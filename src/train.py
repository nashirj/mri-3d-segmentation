"""Helper file for training related functions."""
from sklearn.model_selection import KFold
import torch

from src import evaluate, models

def train_model(network, trainloader, num_epochs, loss_function, optimizer):
    losses = []
    # TODO(): Update this to calculate dice and hd score
    dice_scores = []
    hd_scores = []
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = network(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            losses.append(loss.item())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

        # TODO(): Compute scores here after each epoch
        # dice_scores.append(dice_score)
        # hd_scores.append(hd_score)
    return {
        'losses': losses,
        'dice_scores': dice_scores,
        'hd_scores': hd_scores
    }

def train_kfold(dataset, k_folds, num_epochs, batch_size, loss_function,
                optimizer_type, model_name, model_type, lr):
    """Five fold cross validation"""
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # Dictionary of metrics for each fold, keyed by fold number
    metrics = {}

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
            dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network, train one per fold
        network = model_type()
        network.apply(models.reset_weights)

        # Initialize optimizer
        optimizer = optimizer_type(network.parameters(), lr=lr)

        # Nested dictionary of losses, dice scores, hausdorff scores
        metrics[fold] = train_model(network, trainloader, num_epochs, loss_function, optimizer)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = f'./models/{model_name}-fold-{fold + 1}.pth'
        torch.save(network.state_dict(), save_path)

        # Print about testing
        print('Starting testing')

        correct, total = evaluate.evaluate_model(testloader, network)

        # Print accuracy
        acc = 100.0 * correct / total
        print('Accuracy for fold %d: %d %%' % (fold + 1, acc))
        print('--------------------------------')
        metrics[fold]['accuracy'] = acc
    
    return metrics
