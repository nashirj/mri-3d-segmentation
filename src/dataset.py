"""Five fold cross validation based on:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
"""

import json
import os

import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from . import simpleconvnet

from src import evaluate, train, preprocess


def load_mnist():
    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    dataset_train_part = MNIST(os.getcwd() + "/data", download=True,
                            transform=transforms.ToTensor(), train=True)
    dataset_test_part = MNIST(os.getcwd() + "/data", download=True,
                            transform=transforms.ToTensor(), train=False)
    dataset = ConcatDataset([dataset_train_part, dataset_test_part])

    return dataset


class BratsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, transform=None):
        if data_dir is None:
            data_dir = os.getcwd() + "/data/Task01_BrainTumour"
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'imagesTr')
        self.label_dir = os.path.join(data_dir, 'labelsTr')
        self.transform = transform
        self.img_paths = os.listdir(os.path.join(data_dir, 'imagesTr'))
        self.label_paths = os.listdir(os.path.join(data_dir, 'labelsTr'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_dir + '/' + self.img_paths[idx]
        label_name = self.label_dir + '/' + self.label_paths[idx]
        image = preprocess.load_stacked_mri_img(img_name)
        raw_mask = preprocess.load_mask(label_name)
        mask = preprocess.get_masks(raw_mask)
        
        image = preprocess.normalize_img(image)

        return image, mask


def load_brats():
    dataset = BratsDataset()
    return dataset


def test_kfold():
    # Configuration options
    k_folds = 5
    num_epochs = 1
    batch_size = 10
    loss_function = nn.CrossEntropyLoss()
    optimizer_type = torch.optim.Adam
    model_name = 'simple-convnet-mnist'
    model_type = simpleconvnet.SimpleConvNet
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



if __name__ == '__main__':
    # test_kfold()

    dataset = load_brats()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    import numpy as np
    for i, data in enumerate(dataloader, 0):
        # Get inputs
        inputs, targets = data

        print(inputs.shape)
        print(targets.shape)

        break

