import os.path
import pandas as pd

from core.bert.dataset import DataSet

""" 用于转换数据格式 """

project_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = project_path + "/data/"
train_data = DataSet(dataset_path).get_data(dataset_type="train")
valid_data = DataSet(dataset_path).get_data(dataset_type="valid")
test_data = DataSet(dataset_path).get_data(dataset_type="test")
train_size = len(train_data["label"])
valid_size = len(valid_data["label"])
test_size = len(test_data["label"])
print("train dataset size: {}, valid dataset size: {}, test dataset size: {}".format(train_size, valid_size, test_size))

# train_data_list = [[i, train_data["text"][i], train_data["label"][i]] for i in range(train_size)]
# df = pd.DataFrame(train_data_list, columns=["", "sentence", "label"])
# df.to_excel(dataset_path + "train_and_valid.xlsx", index=False)

valid_data_list = [[i, valid_data["text"][i], valid_data["label"][i]] for i in range(valid_size)]
df = pd.DataFrame(valid_data_list, columns=["", "sentence", "label"])
df.to_excel(dataset_path + "valid.xlsx", index=False)
