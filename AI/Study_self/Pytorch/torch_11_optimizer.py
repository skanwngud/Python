import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device is {}'.format(device))

batch_size = 32
epochs = 30

train_data = torchvision.datasets.MNIST(
    root='data',
    download=False,
    train=True,
    transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.MNIST(
    root='data',
    download=False,
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)

model = Net().to(device)
model.apply(weight_init)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-2
)

def train(model, train_loader, optimizer):
    model.train()

    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    print(f'Train Epoch : {epoch} \tTrain Loss : {loss:.4f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy

for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer)
    test_loss, accuracy = test(model, test_loader)
    print(f'Test Loss : {test_loss:.4f} \tAccuracy : {accuracy}\n')


"""
Train Epoch : 1         Train Loss : 0.0707
Test Loss : 0.0053      Accuracy : 94.5

Train Epoch : 2         Train Loss : 0.0270
Test Loss : 0.0038      Accuracy : 96.29

Train Epoch : 3         Train Loss : 0.1106
Test Loss : 0.0035      Accuracy : 96.34

Train Epoch : 4         Train Loss : 0.1662
Test Loss : 0.0030      Accuracy : 96.99

Train Epoch : 5         Train Loss : 0.0199
Test Loss : 0.0029      Accuracy : 97.25

...

Train Epoch : 30        Train Loss : 0.0090
Test Loss : 0.0028      Accuracy : 98.12
"""