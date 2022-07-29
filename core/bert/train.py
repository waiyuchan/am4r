import time
import torch
from pytorch_pretrained_bert import BertAdam
from torch import nn
from tqdm import tqdm
from core.bert.model import ClassifyModel
from core.bert.processor import DataUtils
from utils.utils import Utils


def train():
    batch_size = 32
    max_seq_len = 200
    epochs = 4
    vocab_pretrained_model_path = "../../pretrained_model/bert-base-uncased/bert-base-uncased-vocab.txt"
    pretrained_model_path = "../../pretrained_model/bert-base-uncased"
    train_iter, valid_iter, total_train_batch, total_valid_batch = DataUtils(
        pretrained_model_name_or_path=vocab_pretrained_model_path, max_seq_len=max_seq_len, batch_size=batch_size).load_data()
    model = ClassifyModel(pretrained_model_path, label_nums=2, is_lock=True)
    print(model)

    optimizer = BertAdam(model.parameters(), lr=5e-05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss_count, train_accuracy_count, n = 0.0, 0.0, 0
        for step, batch_data in tqdm(enumerate(train_iter), desc="train epoch: {}/{}".format(epoch + 1, epochs),
                                     total=total_train_batch):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segements, batch_labels = batch_data

            logits = model(batch_seqs, batch_seq_masks, batch_seq_segements)
            loss = loss_func(logits, batch_labels)
            loss.backward()
            train_loss_count += loss.item()
            logits = logits.softmax(dim=1)
            train_accuracy_count += (logits.argmax(dim=1) == batch_labels).sum().item()
            n += batch_labels.shape[0]
            optimizer.step()
            optimizer.zero_grad()

        model.eval()

        result = Utils.evaluate_accuracy(valid_iter, model, device, total_valid_batch)
        print('epoch %d, loss %.4f, train acc %.3f, time: %.3f' %
              (epoch + 1, train_loss_count / n, train_accuracy_count / n, (time.time() - start)))
        print(result)

    torch.save(model, "../../trained_model/model_fine_tuned_with_base_bert.bin")


if __name__ == '__main__':
    train()
