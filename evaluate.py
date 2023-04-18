import torch

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
