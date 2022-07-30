from pytorch_pretrained_bert import BertTokenizer, BertConfig

bert_model = ""
my_model = ""

tokenizer = BertTokenizer.from_pretrained(bert_model)
moder_config = BertConfig.fro