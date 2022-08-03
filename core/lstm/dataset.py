import numpy as np
import pandas as pd
import torch.utils.data
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [label for label in df["label"]]
        self.texts = [tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
                      for text in df["sentence"]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


if __name__ == '__main__':
    csv_data = pd.read_csv("data/train.csv")
    df = pd.DataFrame(csv_data)
    print(df)
    Dataset(df).classes()
