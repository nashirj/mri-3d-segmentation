import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src import constants

def show_kfold_plots(stacked_values, val_type='loss'):

    mean = np.mean(stacked_values, axis=0)
    std = np.std(stacked_values, axis=0)

    # Plot using std for error bars
    iterations = [i + 1 for i, _ in enumerate(mean)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title(f'Average {val_type} with std. dev for all folds')
    ax[0].plot(iterations, mean)
    ax[0].fill_between(iterations, mean - std, mean + std, alpha=0.3)
    ax[0].set_xlabel('Image number (one epoch))')
    ax[0].set_ylabel(f'Average {val_type}')
    # Show overlain instead of averaged
    ax[1].set_title('Loss for all folds')
    for fold, _ in enumerate(stacked_values):
        ax[1].plot(iterations, stacked_values[fold], label=f'Fold {fold+1}')
    ax[1].set_xlabel('Image number (one epoch)')
    ax[1].set_ylabel('Average loss')
    ax[1].legend()
    plt.suptitle('Loss for all folds, each value is average over all slices of image i')
    plt.show()

    # # Plot using min and max for error bars
    # fig, ax = plt.subplots()
    # ax.set_title('Average loss with min and max for all folds')
    # ax.plot(iterations, mean)
    # ax.fill_between(iterations, mins, maxs, alpha=0.3)
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Loss')
    # plt.show()

def show_kfold_dice_plots(metrics):
    stacked_dice = np.vstack([np.array(v['dice_scores']) for k, v in metrics.items()])
    mean = np.mean(stacked_dice, axis=0)
    std = np.std(stacked_dice, axis=0)
    mins = np.min(stacked_dice, axis=0)
    maxs = np.max(stacked_dice, axis=0)

    # Plot using std for error bars
    iterations = [i + 1 for i, _ in enumerate(mean)]
    fig, ax = plt.subplots()
    ax.set_title('Average dice score with std. dev for all folds')
    ax.plot(iterations, mean)
    ax.fill_between(iterations, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel('Iterations (slice number)')
    ax.set_ylabel('Dice score')
    plt.show()


def plot_values(values, title, xlabel, ylabel, show=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show:
        plt.show()

def show_segmentation_metrics(metrics):
    # Show loss plots
    loss_epoch_0 = metrics['losses'][0]
    plot_values(loss_epoch_0, 'Losses', 'Iterations', 'Loss')

    dice_epoch_0 = metrics['dice_scores'][0]
    # Show dice plots
    plot_values(dice_epoch_0, 'Dice scores', 'Iterations', 'Dice score')

    hd_epoch_0 = metrics['hd_scores'][0]
    # Show hausdorff plots
    plot_values(hd_epoch_0, 'Hausdorff distances, skipping empty masks', 'Iterations', 'Hausdorff distance', show=False)

    hd_full_epoch_0 = metrics['hd_scores_full'][0]
    # Show hausdorff plots
    plot_values(hd_full_epoch_0, 'Hausdorff distances, using nonzero for empty masks', 'Iterations', 'Hausdorff distance')

    


def show_mask(data, ax, label_name=None):
    values = np.unique(data.ravel())
    im = ax.imshow(data)

    if len(values) == 1:
        # Only background
        ax.axis('off')
        return

    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    if not label_name:
        patches = [mpatches.Patch(color=colors[i], label=f"{constants.MASK_TYPES[values[i]]}") for i in range(len(values))]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    else:
        patches = [mpatches.Patch(color=colors[1], label=label_name)]

    ax.axis('off')

def show_aggregate_masks(aggregate_masks, slice_idx, orig_mask):
    # Show perspective 3 with WT, ET, TC
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

    ax[0].set_title(constants.AGGREGATE_MASK_TYPES[constants.ET])
    show_mask(aggregate_masks[constants.ET, :, :, slice_idx], ax[0], label_name=constants.AGGREGATE_MASK_TYPES[constants.ET])

    ax[1].set_title(constants.AGGREGATE_MASK_TYPES[constants.TC])
    show_mask(aggregate_masks[constants.TC, :, :, slice_idx], ax[1], label_name=constants.AGGREGATE_MASK_TYPES[constants.TC])

    ax[2].set_title(constants.AGGREGATE_MASK_TYPES[constants.WT])
    show_mask(aggregate_masks[constants.WT, :, :, slice_idx], ax[2], label_name=constants.AGGREGATE_MASK_TYPES[constants.WT])

    ax[3].set_title('Original mask')
    show_mask(orig_mask[:, :, slice_idx], ax[3])

    plt.show()


if __name__ == '__main__':
    import json
    with open('metrics/unet2d.json', 'r') as f:
        unet2d_metrics = json.load(f)
    

    min_loss_len = min([len(v['losses']) for fold, v in unet2d_metrics.items()])
    min_dice_len = min([len(v['dice_scores']) for fold, v in unet2d_metrics.items()])
    min_hd_len = min([len(v['hd_scores']) for fold, v in unet2d_metrics.items()])
    min_hd_full_len = min([len(v['hd_scores_full']) for fold, v in unet2d_metrics.items()])

    # Truncate all lists to min length
    for fold, v in unet2d_metrics.items():
        v['losses'] = v['losses'][:min_loss_len]
        v['dice_scores'] = v['dice_scores'][:min_dice_len]
        v['hd_scores'] = v['hd_scores'][:min_hd_len]
        v['hd_scores_full'] = v['hd_scores_full'][:min_hd_full_len]

    # Convert string key to int key
    metrics = {int(k): v for k, v in unet2d_metrics.items()}

    # Compute average loss per image for each fold. Each list corresponds to one image
    stacked_losses = np.vstack([np.mean(np.array(v['losses']), axis=1) for _, v in metrics.items()])
    show_kfold_plots(stacked_losses)

    stacked_dice = np.vstack([np.mean(np.array(v['dice_scores']), axis=1) for _, v in metrics.items()])
    show_kfold_plots(stacked_dice, 'dice')
