"""Helper file for training related functions."""
import json

from sklearn.model_selection import KFold
import torch

from src import evaluate, losses, dataset
from src.unet2d import UNet, ActivationFunction, NormalizationLayer, ConvMode, Dimensions, UpMode, create_2d_unet


# def train_3d_model(model, trainloader, num_epochs, loss_function, optimizer, device='cpu'):
#     model = model.to(device)
#     loss_function = loss_function.to(device)

#     losses = []
#     # TODO(): Update this to calculate dice and hd score
#     dice_scores = []
#     hd_scores = []
#     # Run the training loop for defined number of epochs
#     for epoch in range(0, num_epochs):
#         # Print epoch
#         print(f'Starting epoch {epoch+1}')

#         # Set current loss value
#         current_loss = 0.0

#         # Iterate over the DataLoader for training data
#         for i, data in enumerate(trainloader, 0):
#             # Get inputs
#             inputs, targets = data[0].float(), data[1].float()
#             inputs, targets = inputs.to(device), targets.to(device)

#             # input_bytes = inputs.nelement() * inputs.element_size()
#             # input_gb = input_bytes / 1024 / 1024 / 1024
#             # print("Input size in GB:", input_gb)
#             # target_bytes = targets.nelement() * targets.element_size()
#             # target_gb = target_bytes / 1024 / 1024 / 1024
#             # print("Target size in GB:", target_gb)

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB:", cuda_gb_used)
            
#             # Zero the gradients
#             optimizer.zero_grad()

#             # Perform forward pass
#             outputs = model(inputs)

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB after forward:", cuda_gb_used)

#             # # Free up memory
#             # inputs = inputs.to('cpu')
#             # inputs = None
#             # torch.cuda.empty_cache()

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB after clearing inputs:", cuda_gb_used)

#             # Compute loss
#             loss = loss_function(outputs, targets)
#             losses.append(loss.item())

#             # Perform backward pass
#             loss.backward()

#             # Perform optimization
#             optimizer.step()

#             # Print statistics
#             current_loss += loss.item()
#             # if i % 500 == 499:
#             print('Loss after mini-batch %5d: %.3f' %
#                 (i + 1, current_loss / 500))
#             current_loss = 0.0

#     return {
#         'losses': losses,
#         'dice_scores': dice_scores,r, num_epochs, loss_function, optimizer, device='cpu'):
#     model = model.to(device)
#     loss_function = loss_function.to(device)

#     losses = []
#     # TODO(): Update this to calculate dice and hd score
#     dice_scores = []
#     hd_scores = []
#     # Run the training loop for defined number of epochs
#     for epoch in range(0, num_epochs):
#         # Print epoch
#         print(f'Starting epoch {epoch+1}')

#         # Set current loss value
#         current_loss = 0.0

#         # Iterate over the DataLoader for training data
#         for i, data in enumerate(trainloader, 0):
#             # Get inputs
#             inputs, targets = data[0].float(), data[1].float()
#             inputs, targets = inputs.to(device), targets.to(device)

#             # input_bytes = inputs.nelement() * inputs.element_size()
#             # input_gb = input_bytes / 1024 / 1024 / 1024
#             # print("Input size in GB:", input_gb)
#             # target_bytes = targets.nelement() * targets.element_size()
#             # target_gb = target_bytes / 1024 / 1024 / 1024
#             # print("Target size in GB:", target_gb)

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB:", cuda_gb_used)
            
#             # Zero the gradients
#             optimizer.zero_grad()

#             # Perform forward pass
#             outputs = model(inputs)

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB after forward:", cuda_gb_used)

#             # # Free up memory
#             # inputs = inputs.to('cpu')
#             # inputs = None
#             # torch.cuda.empty_cache()

#             # cuda_bytes_used = torch.cuda.memory_allocated(device)
#             # cuda_gb_used = cuda_bytes_used / 1024 / 1024 / 1024
#             # print("CUDA memory used in GB after clearing inputs:", cuda_gb_used)

#             # Compute loss
#             loss = loss_function(outputs, targets)
#             losses.append(loss.item())

#             # Perform backward pass
#             loss.backward()

#             # Perform optimization
#             optimizer.step()

#             # Print statistics
#             current_loss += loss.item()
#             # if i % 500 == 499:
#             print('Loss after mini-batch %5d: %.3f' %
#                 (i + 1, current_loss / 500))
#             current_loss = 0.0

#     return {
#         'losses': losses,
#         'dice_scores': dice_scores,
#         'hd_scores': hd_scores
#     }
#         'hd_scores': hd_scores
#     }

def train_2d_model(model, trainloader, num_epochs, loss_function, optimizer, lr_scheduler, device='cpu', print_freq=50):
    model = model.to(device)
    loss_function = loss_function.to(device)

    losses = []
    # TODO(): Update this to calculate dice and hd score
    dice_scores = []
    hd_scores = []
    hd_scores_full = []

    n_images = len(trainloader.dataset)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)

            assert inputs.shape[0] == 1, "Batch size must be 1 for 2D training"

            img_loss = []
            img_dice = []
            # Skips when only one of the empty masks
            img_hd = []
            # Uses num nonzero elements when one of the masks is empty
            img_hd_full = []
            # Iterate over each slice of the input and target
            for j in range(inputs.shape[4]):
                # Get the current slice
                input_slice = inputs[:, :, :, :, j]
                target_slice = targets[:, :, :, :, j]

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                output_slice = model(input_slice)

                # Compute loss
                loss = loss_function(output_slice, target_slice)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                target_slice = target_slice.detach().cpu().numpy().squeeze(0)
                output_slice = output_slice.detach().cpu().numpy().squeeze(0)

                # Compute metrics
                img_loss.append(loss.item())
                dice, hd, num_nonzero = evaluate.compute_dice_and_hd(output_slice, target_slice)
                img_dice.append(dice)
                if num_nonzero is None:
                    img_hd.append(hd)
                    img_hd_full.append(hd)
                else:
                    img_hd_full.append(num_nonzero)

                # Print statistics
                if j % print_freq == print_freq - 1:
                    print(f"Slice {j+1}, Loss: {loss.item()}, Dice: {dice}, HD: {hd}")

            print(f"finished img {i+1} of {n_images}")

            # Step after each image
            lr_scheduler.step()

            # After each image, compute loss, dice, hd
            losses.append(img_loss)
            dice_scores.append(img_dice)
            hd_scores.append(img_hd)
            hd_scores_full.append(img_hd_full)

            # Cut down on training time
            if i == 200:
                break

        print(f"finished epoch {epoch+1} of {num_epochs}")
        print()
    return {
        'tr_losses': losses,
        'tr_dice_scores': dice_scores,
        'tr_hd_scores': hd_scores,
        'tr_hd_scores_full': hd_scores_full
    }

def train_kfold(dataset, k_folds, num_epochs, batch_size, loss_function,
                optimizer_type, create_model_fn, model_name, lr, device='cpu'):
    """Five fold cross validation"""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=86)

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

        # Init the model, train one per fold
        model = create_model_fn()
        model = model.to(device)

        # Initialize optimizer
        optimizer = optimizer_type(model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Nested dictionary of losses, dice scores, hausdorff scores
        train_metrics = train_2d_model(model, trainloader, num_epochs, loss_function, optimizer, lr_scheduler, device)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model and metrics
        save_path = f'./models/third-run-unet2d/{model_name}-fold-{fold + 1}.pth'
        torch.save(model.state_dict(), save_path)

        # Print about testing
        print('Starting testing')

        test_metrics = evaluate.evaluate_model(model, testloader, device)

        metrics[fold] = train_metrics | test_metrics

        # Save metrics for each fold as we train
        with open(f'./metrics/{model_name}-fold-{fold + 1}.json', 'w') as f:
            json.dump(metrics[fold], f)

    return metrics


if __name__ == '__main__':
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # # Size in gigabytes
    # size_all_gb = size_all_mb / 1024
    # print(f'Model size: {size_all_gb} GB')

    # Configuration options
    k_folds = 5
    num_epochs = 1
    batch_size = 1
    loss_function = losses.BCEDiceLoss(alpha=1, beta=1)
    optimizer_type = torch.optim.Adam
    model_name = 'unet2d'
    lr = 0.02
    dataset = dataset.load_brats()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    metrics = train_kfold(dataset, k_folds, num_epochs, batch_size,
                          loss_function, optimizer_type, create_2d_unet,
                          model_name, lr, device)

    # create json object from dictionary
    js = json.dumps(metrics)

    with open(f"metrics/{model_name}.json", "w") as f:
        f.write(js)

    # evaluate.evaluate_kfold_metrics(k_folds, metrics)
