""" File with CNN models. Add your custom CNN model here. """
import torch
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


class PreTrainedVGG(nn.Module):

    def __init__(self, num_classes=10):
        super(PreTrainedVGG, self).__init__()
        self.vggnet = models.vgg16(pretrained=True)
        num_ftrs = self.vggnet.classifier[6].in_features
        self.vggnet.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vggnet(x)

# 11Million params
class FeatExtResnet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.extract_features = extract_features
        self.disable_gradients(self.resnet)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        # Iterate over model parameters and disable requires_grad
        # This is how we "freeze" these layers (their weights do no change during training)
        if self.extract_features:
            for param in model.parameters():
                param.requires_grad = False

    def param_to_train(self):
        """
                Feature extraction
                Args:
                Returns:
                    None
        """
        params_to_update = []
        if self.extract_features:
            for name, param in self.resnet.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.resnet.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.resnet(x)


class FeatExtDenseNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtDenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.extract_features = extract_features
        self.disable_gradients(self.densenet)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)
        # self.densenet.classifier = nn.Sequential(nn.Linear(1024, 256),
        #                                          nn.ReLU(),
        #                                          nn.Dropout(0.2),
        #                                          nn.Linear(256, 10),
        #                                          nn.LogSoftmax(dim=1))

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        # Iterate over model parameters and disable requires_grad
        # This is how we "freeze" these layers (their weights do no change during training)
        if self.extract_features:
            for param in model.parameters():
                param.requires_grad = False

    def param_to_train(self):
        """
                Feature extraction
                Args:
                Returns:
                    None
        """
        params_to_update = []
        if self.extract_features:
            for name, param in self.densenet.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.densenet.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.densenet(x)

# 128 million params Lol.NO
# class FeatExtVGG(nn.Module):
#     def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
#         super(FeatExtVGG, self).__init__()
#         self.vgg = models.vgg11_bn(pretrained=True)
#         self.extract_features = extract_features
#         self.disable_gradients(self.vgg)
#         num_ftrs = self.vgg.classifier[6].in_features
#         self.vgg.classifier[6]=nn.Linear(num_ftrs, num_classes)
#
#     def disable_gradients(self, model) -> None:
#         """
#         Freezes the layers of a model
#         Args:
#             model: The model with the layers to freeze
#         Returns:
#             None
#         """
#         # Iterate over model parameters and disable requires_grad
#         # This is how we "freeze" these layers (their weights do no change during training)
#         if self.extract_features:
#             for param in model.parameters():
#                 param.requires_grad = False
#
#     def param_to_train(self):
#         """
#                 Feature extraction
#                 Args:
#                 Returns:
#                     None
#         """
#         params_to_update = []
#         if self.extract_features:
#             for name, param in self.vgg.named_parameters():
#                 if param.requires_grad:
#                     params_to_update.append(param)
#         else:
#             params_to_update = self.vgg.parameters()
#         return params_to_update
#
#     def forward(self, x) -> torch.Tensor:
#         """
#                 Forward pass
#                 Args:
#                     x: data
#                 Returns:
#                     classification
#         """
#         return self.vgg(x)


# 722,496 params
class FeatExtSqueeze(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtSqueeze, self).__init__()
        self.squeeze = models.squeezenet1_1(True)
        self.extract_features = extract_features
        self.disable_gradients(self.squeeze)
        self.squeeze.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.squeeze.num_classes = num_classes

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        # Iterate over model parameters and disable requires_grad
        # This is how we "freeze" these layers (their weights do no change during training)
        if self.extract_features:
            for param in model.parameters():
                param.requires_grad = False

    def param_to_train(self):
        """
                Feature extraction
                Args:
                Returns:
                    None
        """
        params_to_update = []
        if self.extract_features:
            for name, param in self.squeeze.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.squeeze.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.squeeze(x)