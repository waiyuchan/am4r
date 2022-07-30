import pandas as pd


class DataSet:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path + "{}.txt"
        self.dataset = {"label": [], "text": []}
        self.POSITIVE_LABEL, self.NEGATIVE_LABEL = 0, 1

    def get_data(self, dataset_type):
        # Train dataset size: 2420, Valid dataset size: 632, Test dataset size: 888
        dataset = pd.read_table(self.dataset_path.format(dataset_type), sep="\t")
        texts = []
        labels = []
        for item in dataset.itertuples():
            positive_text = "{} {} {} And since {}, {}.".format(
                item.debateTitle, item.debateInfo, item.reason, item.warrant0, item.claim) \
                if item.correctLabelW0orW1 == 0 else "{} {} {} And since {}, {}.". \
                format(item.debateTitle, item.debateInfo, item.reason, item.warrant1, item.claim)
            negative_text = "{} {} {} And since {}, {}.".format(
                item.debateTitle, item.debateInfo, item.reason, item.warrant1, item.claim) \
                if item.correctLabelW0orW1 == 0 else "{} {} {} And since {}, {}.". \
                format(item.debateTitle, item.debateInfo, item.reason, item.warrant0, item.claim)
            texts.append(positive_text)
            labels.append(0)
            texts.append(negative_text)
            labels.append(1)
        self.dataset["label"] = labels
        self.dataset["text"] = texts
        return self.dataset

# if __name__ == '__main__':
#     train_data = DataSet().get_data(dataset_type="train")
#     valid_data = DataSet().get_data(dataset_type="valid")
#     test_data = DataSet().get_data(dataset_type="test")
#     print(pd.DataFrame(train_data))
