from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


class ClassifyModel(nn.Module):

    def __init__(self, pretrained_model_name_or_path, label_nums, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_model_name_or_path)
        config = self.bert_model.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, label_nums)
        if is_lock:
            for name, param in self.bert_model.named_parameters():
                if name.startswith("pooler"):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
