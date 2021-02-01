"""CategoricalOp module which the optimizer can customize."""

import torch.nn as nn
from primitives import ConvBnActivation, Identity


class CategoricalOp(nn.Module):

    def __init__(self, op: str = 'conv1x1', activation: str = 'relu', *args, **kwargs):
        super(CategoricalOp, self).__init__(*args, **kwargs)

        if activation == 'relu':
            activation_module = nn.ReLU
        # START TODO #################
        # Handle prelu and gelu as well
        # elif ...
        elif activation == 'prelu':
            activation_module = nn.PReLU
        elif activation == 'gelu':
            activation_module = nn.GELU
        else:  # incase of tanh/sigmoid
            raise NotImplementedError
        # END TODO #################

        if op == 'conv1x1':
            self.op = ConvBnActivation((1, 1), activation_module)
        # START TODO #################
        # handle conv3x3, conv5x5, max3x3 and identity as well
        # For the convolutions, make sure you use the ConvBnActivation module found in lib.primitives
        # For max3x3, use a stride of 1 and the correct padding so that the output has
        # the same dimensions as the input
        elif op == 'conv3x3':
            self.op = ConvBnActivation((3, 3), activation_module)
        elif op == 'conv5x5':
            self.op = ConvBnActivation((5, 5), activation_module)
        elif op == 'max3x3':  # <---- OH revist
            self.op = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        elif op == 'identity':
            self.op = Identity()
        else:
            raise NotImplementedError

        # END TODO #################

    def forward(self, x):
        return self.op(x)
