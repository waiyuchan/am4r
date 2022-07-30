import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm


class Utils:

    def __init__(self):
        pass

    @classmethod
    def evaluate_accuracy(cls, data_iter, net, device, batch_nums):
        predicted_labels, true_labels = [], []

        with torch.no_grad():
            for batch_data in tqdm(data_iter, desc="eval", total=batch_nums):
                batch_data = tuple(t.to(device) for t in batch_data)
                labels = batch_data[-1]
                output = net(*batch_data[:-1])
                predictions = output.softmax(dim=1).argmax(dim=1)
                predicted_labels.append(predictions.detach().cpu().numpy())
                true_labels.append(labels.detach().cpu().numpy())

        return classification_report(np.concatenate(true_labels), np.concatenate(predicted_labels))
