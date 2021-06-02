import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')

train_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=False,
    transform=transforms.Compose(
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
)

test_data = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=transforms.Compose(
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size = 32
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = 32
)