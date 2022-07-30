import math
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

from core.bert.dataset import DataSet


class DataProcessor(object):

    def __init__(self, bert_tokenizer, max_workers=10):
        """

        :param bert_tokenizer: Bert分词器
        :param max_workers:    包含text和label的list数据
        """
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len=30):
        sentences = dataset.iloc[:, 1].tolist()
        labels = dataset.iloc[:, 2].tolist()
        token_seq = list(self.pool.map(self.bert_tokenizer.tokenize, sentences))
        result = list(self.pool.map(self.trunate_and_pad, token_seq, [max_seq_len] * len(token_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        t_seqs = torch.tensor(seqs, dtype=torch.long)
        t_seq_masks = torch.tensor(seq_masks, dtype=torch.long)
        t_seq_segments = torch.tensor(seq_segments, dtype=torch.long)
        t_labels = torch.tensor(labels, dtype=torch.long)

        return TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)

    def trunate_and_pad(self, seq, max_seq_len):
        """
        对超长序列文本进行截断处理
        :param seq:
        :param max_seq_len: 最长序列
        :return:
        """
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]
            # 添加特殊字符
        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)  # id化
        padding = [0] * (max_seq_len - len(seq))  # 根据max_seq_len与seq的长度产生填充序列
        seq_mask = [1] * len(seq) + padding  # 创建seq_mask
        seq_segment = [0] * len(seq) + padding  # 创建seq_segment
        seq += padding  # 对seq拼接填充序列
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


class DataUtils:

    def __init__(self, pretrained_model_name_or_path, max_seq_len, batch_size):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.raw_train_data = pd.read_excel("../../data/train_and_valid.xlsx", sheet_name="train")
        self.raw_valid_data = pd.read_excel("../../data/train_and_valid.xlsx", sheet_name="valid")
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            do_lower_case=True)
        self.processor = DataProcessor(bert_tokenizer=self.bert_tokenizer)
        self.train_data = self.processor.get_input(self.raw_train_data, self.max_seq_len)
        self.valid_data = self.processor.get_input(self.raw_valid_data, self.max_seq_len)

    def load_data(self):
        train_iter = DataLoader(dataset=self.train_data, batch_size=self.batch_size)
        valid_iter = DataLoader(dataset=self.valid_data, batch_size=self.batch_size)

        total_train_batch = math.ceil(len(self.raw_train_data) / self.batch_size)
        total_valid_batch = math.ceil(len(self.raw_valid_data) / self.batch_size)

        return train_iter, valid_iter, total_train_batch, total_valid_batch


if __name__ == '__main__':
    batch_size = 32
    max_seq_len = 200
    pretrained_model_path = "../../pretrained_model/bert-base-uncased/bert-base-uncased-vocab.txt"
    train_iter, valid_iter, total_train_batch, total_valid_batch = DataUtils(
        pretrained_model_name_or_path=pretrained_model_path, max_seq_len=max_seq_len, batch_size=batch_size).load_data()
