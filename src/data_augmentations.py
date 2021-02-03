from torchvision import datasets, transforms
import torch.nn as nn

# Augmentations for Sample Model
resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

# Augmentations for Pretrained model 224 i/p
pretrain_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.5, contrast=0.1),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

validation_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])


# Augmentations for EfficientnetB4 model 380 i/p
efficientnet_augmentations = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.5, contrast=0.1),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375])
])

# B4 needs 380
efficientnet_augmentations_valid = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),])

