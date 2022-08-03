import pandas as pd
import torch.utils.data
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import Dataset
from model import BertClassifier

model_save_path = "models/"


def train(model, train_data, valid_data, learning_rate, epochs):
    train, valid = Dataset(train_data), Dataset(valid_data)
    train_loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=8, shuffle=True)

    # 判断是否使用GPU
    cuda_status = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_status else "cpu")

    # 定义损失函数以及优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if cuda_status:
        model = model.cuda()
        loss_function = loss_function.cuda()

    for epoch in range(epochs):
        # 定义训练集的准确率和损失率
        accuracy_of_train, loss_of_train = 0, 0
        for train_item, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            mask = train_item["attention_mask"].to(device)
            input_id = train_item["input_ids"].squeeze(1).to(device)

            # 通过模型获得输出
            output = model(input_id, mask)

            # 计算损失值
            batch_loss = loss_function(output, train_label)
            loss_of_train += batch_loss.item()

            # 计算准确率
            accuracy = (output.argmax(dim=1) == train_label).sum().item()
            accuracy_of_train += accuracy

            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # 模型验证
        accuracy_of_valid, loss_of_valid = 0, 0
        # 不需要计算梯度
        with torch.no_grad():
            for valid_item, valid_label in valid_loader:
                valid_label = valid_label.to(device)
                mask = valid_item["attention_mask"].to(device)
                input_id = valid_item["input_ids"].squeeze(1).to(device)

                # 通过模型获得输出
                output = model(input_id, mask)

                # 计算损失值
                batch_loss = loss_function(output, valid_label)
                loss_of_valid += batch_loss.item()

                # 计算准确率
                accuracy = (output.argmax(dim=1) == valid_label).sum().item()
                accuracy_of_valid += accuracy

        print(f'''Epochs: {epoch + 1} 
              | Train Loss: {loss_of_train / len(train_data): .3f} 
              | Train Accuracy: {accuracy_of_train / len(train_data): .3f} 
              | Val Loss: {loss_of_valid / len(valid_data): .3f} 
              | Val Accuracy: {accuracy_of_valid / len(valid_data): .3f}''')

        model_name = "model_with_base_bert_{}_{}_{}_{}_{}.pt".format(epoch + 1,
                                                                     '%.3f' % (loss_of_train / len(train_data)),
                                                                     '%.3f' % (accuracy_of_train / len(train_data)),
                                                                     '%.3f' % (loss_of_valid / len(valid_data)),
                                                                     '%.3f' % (accuracy_of_valid / len(valid_data)))

        torch.save(model.state_dict(), model_save_path + model_name)


if __name__ == '__main__':
    epochs = 5
    learning_rate = 1e-6
    model = BertClassifier()

    train_data = pd.DataFrame(pd.read_csv("data/train.csv"))
    valid_data = pd.DataFrame(pd.read_csv("data/valid.csv"))

    train(model, train_data, valid_data, learning_rate, epochs)
