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

pretrain_augmentations_alt = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.Normalize(mean=[0.4920, 0.4749, 0.3117],
                                                                      std=[0.2767, 0.2526, 0.2484])])

# val_transform = transforms.Compose([
#         transforms.Resize(args.test_image_size),
#         transforms.CenterCrop(args.test_crop_image_size),
#         transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4920, 0.4749, 0.3117],
#                          std=[0.2767, 0.2526, 0.2484])
#     ])

efficientnet_augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.5, contrast=0.1),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375])
])

inception_augmentations = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.5, contrast=0.1),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375])
])
