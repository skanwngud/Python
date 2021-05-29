import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device is {}'.format(device))

batch_size = 32
epochs = 30

# data
train_data = torchvision.datasets.MNIST(
    root = 'data',
    train = True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.MNIST(
    root = 'data',
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size
)

# modeling
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout_prob = 0.5

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

# define model, optimizer, criterion
model = Net().to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum=0.5
)
criterion = nn.CrossEntropyLoss()
print(model)

# train, evaluate
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx/len(train_loader):.0f})%] \tTrain Loss : {loss.item():.4f}')
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

for epoch in range(1, epochs + 1):
    print(f'Epochs : {epoch}\n')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = evaluate(model, test_loader)
    print(f'\nTest Loss : {test_loss:.4f} \tAccuracy : {accuracy}\n')

"""
Epochs : 1

Train Epoch : 1 [0/60000 (0)%]  Train Loss : 2.3054
Train Epoch : 1 [6400/60000 (11)%]      Train Loss : 2.0541
Train Epoch : 1 [12800/60000 (21)%]     Train Loss : 1.3528
Train Epoch : 1 [19200/60000 (32)%]     Train Loss : 0.8053
Train Epoch : 1 [25600/60000 (43)%]     Train Loss : 0.5022
Train Epoch : 1 [32000/60000 (53)%]     Train Loss : 0.5269
Train Epoch : 1 [38400/60000 (64)%]     Train Loss : 0.4038
Train Epoch : 1 [44800/60000 (75)%]     Train Loss : 0.5060
Train Epoch : 1 [51200/60000 (85)%]     Train Loss : 0.5032
Train Epoch : 1 [57600/60000 (96)%]     Train Loss : 0.3294

Test Loss : 0.0101      Accuracy : 90.5

Epochs : 2

Train Epoch : 2 [0/60000 (0)%]  Train Loss : 0.3806
Train Epoch : 2 [6400/60000 (11)%]      Train Loss : 0.4232
Train Epoch : 2 [12800/60000 (21)%]     Train Loss : 0.2917
Train Epoch : 2 [19200/60000 (32)%]     Train Loss : 0.3600
Train Epoch : 2 [25600/60000 (43)%]     Train Loss : 0.2733
Train Epoch : 2 [32000/60000 (53)%]     Train Loss : 0.3939
Train Epoch : 2 [38400/60000 (64)%]     Train Loss : 0.2322
Train Epoch : 2 [44800/60000 (75)%]     Train Loss : 0.3945
Train Epoch : 2 [51200/60000 (85)%]     Train Loss : 0.3388
Train Epoch : 2 [57600/60000 (96)%]     Train Loss : 0.2539

Test Loss : 0.0074      Accuracy : 93.03

Epochs : 3

Train Epoch : 3 [0/60000 (0)%]  Train Loss : 0.3289
Train Epoch : 3 [6400/60000 (11)%]      Train Loss : 0.2895
Train Epoch : 3 [12800/60000 (21)%]     Train Loss : 0.2035
Train Epoch : 3 [19200/60000 (32)%]     Train Loss : 0.3075
Train Epoch : 3 [25600/60000 (43)%]     Train Loss : 0.1255
Train Epoch : 3 [32000/60000 (53)%]     Train Loss : 0.4683
Train Epoch : 3 [38400/60000 (64)%]     Train Loss : 0.2221
Train Epoch : 3 [44800/60000 (75)%]     Train Loss : 0.3237
Train Epoch : 3 [51200/60000 (85)%]     Train Loss : 0.1822
Train Epoch : 3 [57600/60000 (96)%]     Train Loss : 0.1993

Test Loss : 0.0057      Accuracy : 94.62

Epochs : 4

Train Epoch : 4 [0/60000 (0)%]  Train Loss : 0.1972
Train Epoch : 4 [6400/60000 (11)%]      Train Loss : 0.2431
Train Epoch : 4 [12800/60000 (21)%]     Train Loss : 0.2177
Train Epoch : 4 [19200/60000 (32)%]     Train Loss : 0.2574
Train Epoch : 4 [25600/60000 (43)%]     Train Loss : 0.1319
Train Epoch : 4 [32000/60000 (53)%]     Train Loss : 0.4528
Train Epoch : 4 [38400/60000 (64)%]     Train Loss : 0.2422
Train Epoch : 4 [44800/60000 (75)%]     Train Loss : 0.2345
Train Epoch : 4 [51200/60000 (85)%]     Train Loss : 0.1019
Train Epoch : 4 [57600/60000 (96)%]     Train Loss : 0.1886

Test Loss : 0.0047      Accuracy : 95.53

Epochs : 5

Train Epoch : 5 [0/60000 (0)%]  Train Loss : 0.1898
Train Epoch : 5 [6400/60000 (11)%]      Train Loss : 0.2218
Train Epoch : 5 [12800/60000 (21)%]     Train Loss : 0.1979
Train Epoch : 5 [19200/60000 (32)%]     Train Loss : 0.1781
Train Epoch : 5 [25600/60000 (43)%]     Train Loss : 0.0562
Train Epoch : 5 [32000/60000 (53)%]     Train Loss : 0.3289
Train Epoch : 5 [38400/60000 (64)%]     Train Loss : 0.2043
Train Epoch : 5 [44800/60000 (75)%]     Train Loss : 0.3039
Train Epoch : 5 [51200/60000 (85)%]     Train Loss : 0.1300
Train Epoch : 5 [57600/60000 (96)%]     Train Loss : 0.2596

Test Loss : 0.0040      Accuracy : 96.07

...

Epochs : 30

Train Epoch : 30 [0/60000 (0)%]         Train Loss : 0.0041
Train Epoch : 30 [6400/60000 (11)%]     Train Loss : 0.0147
Train Epoch : 30 [12800/60000 (21)%]    Train Loss : 0.0470
Train Epoch : 30 [19200/60000 (32)%]    Train Loss : 0.0222
Train Epoch : 30 [25600/60000 (43)%]    Train Loss : 0.0052
Train Epoch : 30 [32000/60000 (53)%]    Train Loss : 0.1200
Train Epoch : 30 [38400/60000 (64)%]    Train Loss : 0.0905
Train Epoch : 30 [44800/60000 (75)%]    Train Loss : 0.0183
Train Epoch : 30 [51200/60000 (85)%]    Train Loss : 0.0043
Train Epoch : 30 [57600/60000 (96)%]    Train Loss : 0.0639

Test Loss : 0.0018      Accuracy : 98.31
"""