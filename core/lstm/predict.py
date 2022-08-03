import pandas as pd
import torch
import torch.utils.data

from dataset import Dataset
from model import BertLSTMClassifier


def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)
    cuda_status = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_status else "cpu")
    if cuda_status:
        model = model.cuda()

    accuracy_of_test = 0
    with torch.no_grad():
        test_hidden = model.init_hidden(batch_size=8)
        for test_item, test_label in test_dataloader:
            test_label = test_label.to(device)
            input_id = test_item['input_ids'].squeeze(1).to(device)
            output = model(input_id, test_hidden)
            accuracy = (output.argmax(dim=1) == test_label).sum().item()
            accuracy_of_test += accuracy

    print(f'Test Accuracy: {accuracy_of_test / len(test_data): .3f}')


if __name__ == '__main__':
    model = BertLSTMClassifier()
    model.load_state_dict(torch.load("models/model_with_base_bert_and_lstm_10_0.001_0.995_0.032_0.947.pt"))
    test_data = pd.DataFrame(pd.read_csv("data/test.csv"))
    evaluate(model, test_data)
