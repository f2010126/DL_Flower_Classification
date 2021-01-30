""" File with CNN models. Add your custom CNN model here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


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

    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
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


class FeatExtSqueeze(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtSqueeze, self).__init__()
        self.squeeze = models.squeezenet1_1(True)
        self.extract_features = extract_features
        self.disable_gradients(self.squeeze)
        self.squeeze.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
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


class FeatExtEfficientNet(nn.Module):

    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtEfficientNet, self).__init__()
        self.efficient = EfficientNet.from_pretrained('efficientnet-b0')
        self.extract_features = extract_features
        # feat ext
        # self.disable_gradients(self.efficient)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.5)
        self.l2 = nn.Linear(256, num_classes)  # 6 is number of classes
        self.relu = nn.LeakyReLU()

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
        else:
            params_to_update = self.efficient.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        x = self.efficient(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


# 5 million
class FeatExtGoogLeNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtGoogLeNet, self).__init__()
        self.google = models.googlenet(pretrained=True)
        self.extract_features = extract_features
        self.disable_gradients(self.google)
        num_ftrs = self.google.fc.in_features
        self.google.fc = nn.Linear(num_ftrs, num_classes)

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
            for name, param in self.google.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.google.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.google(x)


# 24 million
class FeatExtInception(nn.Module):
    def __init__(self, input_shape=(3, 299, 299), num_classes=10, extract_features=True):
        super(FeatExtInception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.extract_features = extract_features
        self.disable_gradients(self.inception)
        num_ftrs = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

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
            for name, param in self.inception.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.inception.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.inception(x)


# 1.26million
class FeatExtShuffleNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, extract_features=True):
        super(FeatExtShuffleNet, self).__init__()
        self.shuffle = models.shufflenet_v2_x1_0(pretrained=True)
        self.extract_features = extract_features
        self.disable_gradients(self.shuffle)
        num_ftrs = self.shuffle.fc.in_features
        self.shuffle.fc = nn.Linear(num_ftrs, num_classes)

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
            for name, param in self.shuffle.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.shuffle.parameters()
        return params_to_update

    def forward(self, x) -> torch.Tensor:
        """
                Forward pass
                Args:
                    x: data
                Returns:
                    classification
        """
        return self.shuffle(x)
