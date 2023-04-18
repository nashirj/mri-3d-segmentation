"""Helper file for training related functions."""

def train_clf(network, trainloader, num_epochs, loss_function, optimizer):
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
    return losses, dice_scores, hd_scores
