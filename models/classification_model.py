
from typing import Tuple

import torch

from transformers import AutoModel, AutoConfig


class ClassificationModel(torch.nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()

        # Main model
        transformers_config = AutoConfig.from_pretrained(config['model_name_or_path'], adapter_size=config['adapter_size'])
        self.transformer = AutoModel.from_pretrained(config['model_name_or_path'], config=transformers_config)
        self.hidden_size = config['hidden_size']
        self.num_labels = num_labels

        classifier_dropout = (
            config['classifier_dropout'] if config.get('classifier_dropout') is not None else 0
        )
        self.dropout = torch.nn.Dropout(classifier_dropout).to(dtype=self.transformer.dtype)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels).to(dtype=self.transformer.dtype)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        
        # get the index of the final representation
        sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1

        # shape : (batch, num_labels)
        pooled_output = last_hidden_state[range(last_hidden_state.size(0)), sequence_lengths]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        predictions = torch.argmax(logits, dim=1).tolist()

        return loss, predictions
