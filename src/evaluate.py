from medpy.metric import dc, hd
import numpy as np
import torch

from src import viz, preprocess


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
    viz.show_kfold_loss_plots(metrics)


def evaluate_segmentation_scores(metrics):
    # Show loss plots
    viz.show_loss_plots(metrics)


def compute_dice_and_hd(y_pred, y_true):
    # Convert to np
    flat_pred = y_pred.flatten()
    flat_true = y_true.flatten()
    # binarize
    flat_pred = preprocess.binarize_mask(flat_pred)
    flat_true = preprocess.binarize_mask(flat_true)

    unique_pred = np.unique(flat_pred)
    unique_true = np.unique(flat_true)

    dice = dc(flat_pred, flat_true)

    # If skip_hd is not None, skip computing hd
    skip_hd = None

    empty_pred = len(unique_pred) == 1 and unique_pred[0] == 0
    empty_true = len(unique_true) == 1 and unique_true[0] == 0
    # Check for empty masks
    if empty_pred and empty_true:
        skip_hd = 0
    if empty_pred:
        # Get number of non-zero elements in true mask
        skip_hd = np.count_nonzero(flat_true)
    if empty_true:
        # Get number of non-zero elements in pred mask
        skip_hd = np.count_nonzero(flat_pred)

    if skip_hd is not None:
        return dice, None, skip_hd
    return dice, hd(flat_pred, flat_true), skip_hd


if __name__ == '__main__':
    # import json

    # # Load metrics from json file
    # with open('metrics/simple-convnet-mnist.json', 'r') as f:
    #     metrics = json.load(f)
    #     metrics = {int(k): v for k, v in metrics.items()}

    # evaluate_kfold_metrics(5, metrics)

    # Test dice and hd
    # segmentation
    seg = np.zeros((100,100,100), dtype='int')
    seg[30:70, 30:70, :] = 1

    # ground truth
    gt = np.zeros((100,100, 100), dtype='int')
    gt[30:70, 40:80, :] = 1

    print(compute_dice_and_hd(seg, gt))
    print()
    print(compute_dice_and_hd(gt, seg))
    print()
    print(compute_dice_and_hd(seg, seg))
    print()
    print(compute_dice_and_hd(gt, gt))
    print()

    # Test dice and hd multiclass
    seg[30:70, 70:90, :] = 2
    gt[30:70, 80:100, :] = 2

    print(compute_dice_and_hd(seg, gt))
    print()

    print("Testing empty masks")

    # Test dice and hd with empty masks
    seg = np.zeros((100,100,100), dtype='int')
    gt = np.zeros((100,100, 100), dtype='int')
    print(compute_dice_and_hd(seg, gt))
    print()

    seg = np.zeros((100,100,100), dtype='int')
    gt = np.ones((100,100, 100), dtype='int')
    print(compute_dice_and_hd(seg, gt))
    print()

    seg = np.ones((100,100,100), dtype='int')
    gt = np.zeros((100,100, 100), dtype='int')
    print(compute_dice_and_hd(seg, gt))

