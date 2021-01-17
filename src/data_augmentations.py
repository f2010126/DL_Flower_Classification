from torchvision import datasets, transforms
import torch.nn as nn

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

resize_to_128x128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

custom_augment = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.5, contrast=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor()
    # add a normalise here
])

