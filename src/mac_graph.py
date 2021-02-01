"""MacroGraph which we want to optimize."""

import torch.nn as nn
import ConfigSpace

from primitives import Stem
from cell import CellA, CellB


class MacroGraph(nn.Module):
    """The MacroGraph, consisting of the CellA and CellB, which we aim to optimize."""

    def __init__(self,
                 config_cell_a: ConfigSpace.ConfigurationSpace,
                 config_cell_b: ConfigSpace.ConfigurationSpace,
                 *args,
                 **kwargs):
        super(MacroGraph, self).__init__(*args, **kwargs)
        self.stem = Stem(10)

        # START TODO #################
        self.a1 = CellA(config_cell_a)
        self.b1 = CellB(config_cell_b)
        self.max3x3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.a2 = CellA(config_cell_a)
        self.b2 = CellB(config_cell_b)
        self.fc = nn.Linear(10*295*295, 10)  # o/p 10 classes <-- check that
        # END TODO #################

    def forward(self, x):
        x = self.stem(x)

        # START TODO #################
        x = self.a1(x)
        x = self.b1(x)
        x = self.max3x3(x)
        x = self.a2(x)
        x = self.b2(x)
        x = self.max3x3(x)
        x = x.view(x.size(0), -1)  # <--- do not forget to flatten
        x = self.fc(x)
        # END TODO #################
        return x

    def param_to_train(self):
        return self.parameters()
