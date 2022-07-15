import logging
import os
import random
import time
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import deepspeed
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    set_seed,
)
from models import ClassificationModel

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    start_time = time.time()

    # To avoid warnings about parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    ds_config = dict(config['ds_configs'])

    # Process auto config
    # auto values are from huggingface
    if ds_config['zero_optimization']['reduce_bucket_size'] == 'auto':
        ds_config['zero_optimization']['reduce_bucket_size'] = config['models']['hidden_size'] * config['models']['hidden_size']
    if ds_config['zero_optimization']['stage3_prefetch_bucket_size'] == 'auto':
        ds_config['zero_optimization']['stage3_prefetch_bucket_size'] = config['models']['hidden_size'] * config['models']['hidden_size'] * 0.9
    if ds_config['zero_optimization']['stage3_param_persistence_threshold'] == 'auto':
        ds_config['zero_optimization']['stage3_param_persistence_threshold'] = config['models']['hidden_size'] * 10

    # For huggingface deepspeed / Keep this alive!
    dschf = HfDeepSpeedConfig(ds_config)

    # Set logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    # logging level info for main process only
    datasets.disable_progress_bar()
    if local_rank == 0:
        transformers.logging.set_verbosity_info()
        datasets.logging.set_verbosity_info()
        logger.setLevel(logging.INFO)
    else:
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        logger.setLevel(logging.ERROR)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if config['datasets']['benchmark'] is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(config['datasets']['benchmark'], config['datasets']['task'])
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if config['datasets']['train_file'] is not None:
            data_files["train"] = config['datasets']['train_file']
        if config['datasets']['validation_file'] is not None:
            data_files["validation"] = config['datasets']['validation_file']
        extension = (config['datasets']['train_file'] if config['datasets']['train_file'] is not None else config['datasets']['validation_file']).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load model
    logger.info(f'Start loading model {config["models"]["model_name_or_path"]}')
    model_loading_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config['models']["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = ClassificationModel(config["models"], config['datasets']['num_labels'])

    # Set optimizer / you can also use deepspeed config to create optimizer use at your convenience
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Tokenize the texts
        result = {}
        texts = examples[config['datasets']['sentence1_key']] if config['datasets']['sentence2_key'] is None else examples[config['datasets']['sentence1_key']] + '\n' + examples[config['datasets']['sentence2_key']]
        
        result['inputs'] = texts
        # result = tokenizer(*texts, padding=config['padding'], max_length=config['max_length'], truncation=True)
        result["labels"] = examples[config['datasets']['label_key']]

        return result

    # Ensures the main process performs the mapping
    if local_rank > 0:  
        logger.info("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    processed_datasets = raw_datasets.map(
        preprocess_function,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    if local_rank == 0:
        torch.distributed.barrier()

    train_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["validation"]
    batch_size = ds_config['train_micro_batch_size_per_gpu']

    # Evaluate! 
    logger.info("***** Few-shot Evaluation *****")
    logger.info(f"  TASK                                = {config['datasets']['task']}")
    logger.info(f"  Num TRAIN examples                  = {len(train_dataset)}")
    logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
    logger.info(f"  Random Seed                         = {config['seed']}")
    logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    metric = load_metric('accuracy', num_process=world_size, process_id=local_rank)

    for epoch in range(config['epochs']):
        model_engine.module.train()
        logger.info(f"Epoch {epoch}: START Training")
        if local_rank == 0:
            progressbar = tqdm(range(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):
            # to local device
            inputs = tokenizer(batch['inputs'], padding=config['padding'], max_length=config['max_length'], return_tensors='pt').to(device=local_rank)
            inputs['labels'] = batch['labels'].to(device=local_rank)
            # batch = {k:v.to(local_rank) for k, v in batch.items()}
            loss, predictions = model_engine(**inputs)

            model_engine.backward(loss)
            model_engine.step()

            metric.add_batch(predictions=predictions, references=batch['labels'])
            if local_rank == 0:
                progressbar.update(1)

        result = metric.compute()
        if local_rank == 0:
            logger.info(f"Epoch {epoch}: Train accuracy {result['accuracy'] * 100}")

        logger.info(f"Epoch {epoch}: START Evaluation")
        model_engine.module.eval()
        progressbar = tqdm(range(len(test_dataloader)))
        for step, batch in enumerate(test_dataloader):
            inputs = tokenizer(batch['inputs'], padding=config['padding'], max_length=config['max_length'], return_tensors='pt').to(device=local_rank)
            inputs['labels'] = batch['labels'].to(device=local_rank)
            with torch.no_grad():
                loss, predictions = model_engine(**inputs)

            metric.add_batch(predictions=predictions, references=batch['labels'])
            if local_rank == 0:
                progressbar.update(1)
        if local_rank == 0:
            progressbar.close()
        result = metric.compute()
        if local_rank == 0:
            logger.info(f"Epoch {epoch}: Evaluation accuracy {result['accuracy'] * 100}")
        #save checkpoint
        model_engine.save_checkpoint(config['save_path'], epoch)
   
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    main()
    
    