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

    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
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

# 3 million param
class TinyVGG(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes= 10):
        super(TinyVGG, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = self.fc1 = nn.Linear(12544, 128) # <----
        self.norm1d = nn.BatchNorm1d(128) # 1x1 conv :O
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
       x = self.pool(F.relu(self.norm1(self.conv1(x))))
       x = self.pool(F.relu(self.norm2(self.conv2(x))))
       x = self.pool(F.relu(self.norm3(self.conv3(x))))
       x = self.pool(F.relu(self.norm4(self.conv4(x))))
       x = x.view(x.size(0), -1)
       x = F.relu(self.norm1d(self.fc1(x)))
       x = self.fc2(x)
       return x

# smaller vgg 18Mill
class SmallerVGG(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes= 10):
        super(SmallerVGG, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, padding = 1)
        self.conv11 = nn.Conv2d(64, 64, 7 ,padding=1)
        self.conv12 = nn.Conv2d(64, 64, 7, padding=1)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 5, padding = 1)
        self.conv21 = nn.Conv2d(64, 64, 5, padding = 1)
        self.conv22 = nn.Conv2d(64, 64, 5, padding=1)
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv31 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv41 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv42 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv51 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = self.fc1 = nn.Linear(18432, 512) # <----
        self.norm1d = nn.BatchNorm1d(512) # 1x1 conv :O
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.norm1(self.conv12(x))))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.norm2(self.conv22(x))))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv31(x))
        x = self.pool(F.relu(self.norm3(self.conv32(x))))

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv41(x))
        x = self.pool(F.relu(self.norm4(self.conv42(x))))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv51(x))
        x = self.pool(F.relu(self.conv52(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.norm1d(self.fc1(x)))
        x = self.fc2(x)
        return x