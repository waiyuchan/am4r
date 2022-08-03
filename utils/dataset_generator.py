import pandas as pd


def train_and_test_dataset_generator():
    csv_data = pd.read_csv("data/train_amazon.csv")
    df = pd.DataFrame(csv_data)
    data_list = [[item["text"], item["label"]] for column, item in df.iterrows()]
    train_len = int(len(data_list) * 0.3 * 0.1)
    valid_len = int(len(data_list) * 0.3 * 0.12) - int(len(data_list) * 0.3 * 0.1)

    train_true_len = train_len / 2
    train_false_len = train_len / 2
    train_data_list = []
    train_true_count = 0
    train_false_count = 0
    continue_point = 0

    for i in range(0, len(data_list)):
        item = data_list[i]
        if item[1] == 0 and train_true_count < train_true_len:
            train_data_list.append(item)
            train_true_count += 1
        if item[1] == 1 and train_false_count < train_false_len:
            train_data_list.append(item)
            train_false_count += 1
        if train_true_count == train_true_len and train_false_count == train_false_len:
            continue_point = i
            break

    valid_true_len = valid_len / 2
    valid_false_len = valid_len / 2
    valid_data_list = []
    valid_true_count = 0
    valid_false_count = 0

    for i in range(continue_point, len(data_list)):
        item = data_list[i]
        if item[1] == 0 and valid_true_count < valid_true_len:
            valid_data_list.append(item)
            valid_true_count += 1
        if item[1] == 1 and valid_false_count < valid_false_len:
            valid_data_list.append(item)
            valid_false_count += 1
        if valid_true_count == valid_true_len and valid_false_count == valid_false_len:
            break

    train_df = pd.DataFrame(train_data_list, columns=["sentence", "label"])
    train_df.to_csv("data/train.csv", index=False)

    valid_df = pd.DataFrame(valid_data_list, columns=["sentence", "label"])
    valid_df.to_csv("data/valid.csv", index=False)


def test_dataset_generator():
    csv_data = pd.read_csv("data/test_amazon.csv")
    df = pd.DataFrame(csv_data)
    data_list = [[item["text"], item["label"]] for column, item in df.iterrows()]
    test_data_list = data_list[:int(len(data_list) * 0.04)]
    test_df = pd.DataFrame(test_data_list, columns=["sentence", "label"])
    test_df.to_csv("data/test.csv", index=False)


if __name__ == '__main__':
    test_dataset_generator()
