import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.CIFAR10(
    roo = 'data',
    train = False,
    download=True,
    transform=ToTensor(),
)

train_load = DataLoader(
    training_data,
    batch_size=64,
)

test_load = DataLoader(
    test_data,
    batch_size=64
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            print(f'loss : {loss:>0.4f}')

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y, = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).iteml()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size

        print(f'Accuracy : {100 * correct:>0.1f}%, Avg loss : {test_loss:>4f}')

epochs = 10
for t in range(epochs):
    print(f'Epochs : {t + 1}')
    train(train_load, model, loss_fn, optimizer)
    test(test_load, model)