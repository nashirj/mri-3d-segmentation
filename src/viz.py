import matplotlib.pyplot as plt
import numpy as np

def show_loss_plots(metrics):
    stacked_losses = np.vstack([np.array(v['losses']) for k, v in metrics.items()])
    mean = np.mean(stacked_losses, axis=0)
    std = np.std(stacked_losses, axis=0)
    mins = np.min(stacked_losses, axis=0)
    maxs = np.max(stacked_losses, axis=0)

    # Plot using std for error bars
    iterations = [i + 1 for i, _ in enumerate(mean)]
    fig, ax = plt.subplots()
    ax.set_title('Average loss with std. dev for all folds')
    ax.plot(iterations, mean)
    ax.fill_between(iterations, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    plt.show()

    # # Plot using min and max for error bars
    # fig, ax = plt.subplots()
    # ax.set_title('Average loss with min and max for all folds')
    # ax.plot(iterations, mean)
    # ax.fill_between(iterations, mins, maxs, alpha=0.3)
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Loss')
    # plt.show()
