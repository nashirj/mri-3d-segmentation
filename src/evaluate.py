import numpy as np
import torch

from src import viz

def evaluate_model(testloader, network):
    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data

            # Generate outputs
            outputs = network(inputs)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct, total

def evaluate_kfold_metrics(k_folds, metrics):
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    summed_acc = 0.0
    best_acc = 0.0
    best_fold = 0
    for fold_num, values in metrics.items():
        acc = values['accuracy']
        print(f'Fold {fold_num + 1}: {acc:.3f}%')
        summed_acc += acc
        if acc > best_acc:
            best_acc = acc
            best_fold = fold_num + 1

    print(f'Average: {summed_acc/len(metrics.items()):.3f}%')

    print(f"Best performing model is from fold: {best_fold}")

    # Show loss plots
    viz.show_loss_plots(metrics)

if __name__ == '__main__':
    import json

    # Load metrics from json file
    with open('metrics/simple-convnet-mnist.json', 'r') as f:
        metrics = json.load(f)
        metrics = {int(k): v for k, v in metrics.items()}

    evaluate_kfold_metrics(5, metrics)
