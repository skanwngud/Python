from typing import Sequence
import torch

from torch import nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download=True,
    transform=ToTensor()
)

train_load = DataLoader(
    training_data, batch_size=64
)

test_load = DataLoader(
    test_data, batch_size=64
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = Sequential(
            nn.Conv2d(1, 32, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = Sequential(
            nn.Conv2d(32, 64, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fclayer = Sequential(
            nn.Linear(64 * 7 * 7, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fclayer(out)
        return out

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            print(f'loss : {loss:>.4f}')

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f'loss : {test_loss:>.4f}, acc : {correct:>.4f}')

epochs = 10
for t in range(epochs):
    print(f'epochs : {t + 1}')
    train(train_load, model, loss_fn, optimizer)
    test(test_load, model)


import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    download=True,
    train = True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root = 'data',
    download=True,
    train=False,
    transform=ToTensor()
)

train_load = DataLoader(
    train_data, batch_size = 64
)

test_load = DataLoader(
    test_data, batch_size=64
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 10),
            nn.ReLU()
        )

    def foward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def train(dataload, model, loss_fn, optimizer):
    