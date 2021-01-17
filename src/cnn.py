""" File with CNN models. Add your custom CNN model here. """

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        self.fc1 = nn.Linear(in_features=4500, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


# python -m src.main --model SmallCNN --epochs 25
class SmallCNN(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4320, 128)  # <--
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# python -m src.main --model SmallCNN2 --epochs 25
class SmallCNN3(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=10):
        super(SmallCNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=7, padding=0, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3000, 128)  # <--
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# 83k param 12.5% accuracy. :/
class SmallCNN4(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=10):
        super(SmallCNN4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=3, padding=0, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=20, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=120, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3000, 128)  # <--
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PreTrainedCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(PreTrainedCNN, self).__init__()
        self.vggnet = models.vgg16(pretrained=True)
        num_ftrs = self.vggnet.classifier[6].in_features
        self.vggnet.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # just in case
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vggnet(x)
