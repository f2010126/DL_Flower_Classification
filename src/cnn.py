""" File with CNN models. Add your custom CNN model here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import efficientnet_pytorch
# from efficientnet_pytorch import EfficientNet


class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=(1,1))
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

    def param_to_train(self):
        return self.parameters()

class FeatExtEfficientNetB4(nn.Module):

    def __init__(self, input_shape=(3, 380, 380), num_classes=10, extract_features=True):
        super(FeatExtEfficientNetB4, self).__init__()
        self.efficient = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')
        self.extract_features = True
        self.disable_gradients(self.efficient) # freeze model
        self.classifier_layer = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )


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
            for name, param in self.efficient.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
            params_to_update.extend(self.classifier_layer.parameters())
        else:
            params_to_update = self.classifier_layer.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        x = self.efficient.extract_features(x)
        x = self.efficient._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficient._dropout(x)
        x = self.classifier_layer(x)
        return x

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

