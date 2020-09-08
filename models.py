from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

# todo: type hinting
# todo:


class BertForQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQA, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
    ):
        bert_out = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
        )
        seq_out = bert_out[0]
        pooled_out = bert_out[1]

        # predict start & end positions
        qa_logits = self.qa_outputs(seq_out)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits.squeeze(-1)
        end_logits.squeeze(-1)

        # classification
        classifier_logits = self.classifier(self.dropout(pooled_out))

        return start_logits, end_logits, classifier_logits


class ConditionedBertForQA(BertPreTrainedModel):
    def __init__(self, config):
        super(ConditionedBertForQA, self).__init__(config)
        # todo
