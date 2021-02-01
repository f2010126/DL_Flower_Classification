"""Primitive modules used in the MacroGraph and Cells."""

import torch
import torch.nn as nn
import numpy as np


class Stem(nn.Module):
    """Basic convolution with three input channels and C output channels,
       followed by batch normalization."""

    def __init__(self, C):
        super(Stem, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C)
        )

    def forward(self, x, *args, **kwargs):
        return self.seq(x)


class ConvBnActivation(nn.Module):
    """Convolution, followed by batch normalization, followed by relu activation."""

    def __init__(self, kernel_size: int, activation: nn.Module):
        super(ConvBnActivation, self).__init__()
        self.in_channels = 10
        self.out_channels = 10
        self.kernel_size = kernel_size
        self._compute_same_padding()
        self.activation = activation()

        self.seq = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            self.activation
        )

    def forward(self, x, *args, **kwargs):
        return self.seq(x)

    def _compute_same_padding(self):
        """Computes 'same' padding, assuming the stride is 1.

        Returns:
            None
        """
        pad1 = (int(self.kernel_size[0] - 1)) // 2
        pad2 = (int(self.kernel_size[1] - 1)) // 2
        self.padding = (pad1, pad2)


class Identity(nn.Module):
    """Identity module. Doesn't perform any operation."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Sum(nn.Module):
    """Module to sum the inputs to this module. Assumes all the
       inputs have the same number of channels and spatial dimensions."""

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, X):
        return sum(X)
