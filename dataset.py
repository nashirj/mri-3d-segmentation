"""Five fold cross validation based on:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
"""

import os
from sklearn.model_selection import KFold
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

import evaluate
import models
import train


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
    loss_function = nn.CrossEntropyLoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    dataset = load_mnist()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'Training model on FOLD {fold + 1}/{k_folds}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=10, sampler=test_subsampler)

        # Init the neural network, train one per fold
        network = models.SimpleConvNet()
        network.apply(models.reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        # Losses, dice scores, hausdorff scores
        metrics = train.train_model(trainloader, network, loss_function, optimizer, num_epochs)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = f'./model-fold-{fold + 1}.pth'
        torch.save(network.state_dict(), save_path)

        # Print about testing
        print('Starting testing')

        correct, total = evaluate.evaluate_model(testloader, network)

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    print("Best performing model is from fold: " % max(results, key=results.get))
