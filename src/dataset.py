"""Five fold cross validation based on:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
"""

import json
import os

import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from src import evaluate, models, train


def load_mnist():
    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    dataset_train_part = MNIST(os.getcwd() + "/data", download=True,
                            transform=transforms.ToTensor(), train=True)
    dataset_test_part = MNIST(os.getcwd() + "/data", download=True,
                            transform=transforms.ToTensor(), train=False)
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])

    return dataset


if __name__ == '__main__':
    # Configuration options
    k_folds = 5
    num_epochs = 1
    batch_size = 10
    loss_function = nn.CrossEntropyLoss()
    optimizer_type = torch.optim.Adam
    model_name = 'simple-convnet-mnist'
    model_type = models.SimpleConvNet
    lr = 1e-4

    # Set fixed random number seed
    torch.manual_seed(42)

    dataset = load_mnist()

    metrics = train.train_kfold(dataset, k_folds, num_epochs, batch_size,
                                loss_function, optimizer_type, model_name,
                                model_type, lr)

    # create json object from dictionary
    js = json.dumps(metrics)

    with open(f"metrics/{model_name}.json", "w") as f:
        f.write(js)

    evaluate.evaluate_kfold_metrics(k_folds, metrics)
