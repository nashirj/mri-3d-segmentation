import torch.nn as nn

class SimpleConvNet(nn.Module):
    '''Simple Convolutional Neural Network.'''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def reset_weights(m):
    '''Try resetting model weights to avoid weight leakage.'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
