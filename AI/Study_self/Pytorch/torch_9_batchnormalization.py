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
    root='data',
    download=False,
    train=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root='data',
    download=False,
    train=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size = batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = batch_size
)

# modeling
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)                  # fc1 output dim 이 512 이므로 nn.BatchNorm1d 를 512 로 설정
        self.batch_norm2 = nn.BatchNorm1d(256)                  # fc2 output dim 이 256 이므로 nn.BatchNorm1d 를 256 로 설정

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.batch_norm1(x)                                 # fc1 output 을 위에서 정의한 batch_norm1 의 input 으로 정의
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout_prob, training=self.training)
        x = self.fc2(x)
        x = self.batch_norm2(x)                                 # fc2 output 을 위에서 정의한 batch_norm2 의 input 으로 정의
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# train, eval
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 1e-2,
    momentum=0.5
)

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
            print('Train Epoch : {} [{}/{} ({:.0f})%] \tTrain Loss : {:.4f}'.format(
                epoch,
                batch_idx * len(image),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

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

# training
for epoch in range(1, epochs+1):
    print(f'Epochs : {epoch}\n')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = evaluate(model, test_loader)
    print('\nTest Loss : {:.4f} \tAccuracy : {}\n'.format(
        test_loss, accuracy
    ))


"""
Epochs : 1

Train Epoch : 1 [0/60000 (0)%]  Train Loss : 2.5206
Train Epoch : 1 [6400/60000 (11)%]      Train Loss : 0.8108
Train Epoch : 1 [12800/60000 (21)%]     Train Loss : 0.4677
Train Epoch : 1 [19200/60000 (32)%]     Train Loss : 0.5048
Train Epoch : 1 [25600/60000 (43)%]     Train Loss : 0.2968
Train Epoch : 1 [32000/60000 (53)%]     Train Loss : 0.5783
Train Epoch : 1 [38400/60000 (64)%]     Train Loss : 0.4836
Train Epoch : 1 [44800/60000 (75)%]     Train Loss : 0.4599
Train Epoch : 1 [51200/60000 (85)%]     Train Loss : 0.2244
Train Epoch : 1 [57600/60000 (96)%]     Train Loss : 0.3408

Test Loss : 0.0054      Accuracy : 94.64166666666667

Epochs : 2

Train Epoch : 2 [0/60000 (0)%]  Train Loss : 0.2152
Train Epoch : 2 [6400/60000 (11)%]      Train Loss : 0.6386
Train Epoch : 2 [12800/60000 (21)%]     Train Loss : 0.2766
Train Epoch : 2 [19200/60000 (32)%]     Train Loss : 0.0857
Train Epoch : 2 [25600/60000 (43)%]     Train Loss : 0.1648
Train Epoch : 2 [32000/60000 (53)%]     Train Loss : 0.3952
Train Epoch : 2 [38400/60000 (64)%]     Train Loss : 0.3949
Train Epoch : 2 [44800/60000 (75)%]     Train Loss : 0.5031
Train Epoch : 2 [51200/60000 (85)%]     Train Loss : 0.1194
Train Epoch : 2 [57600/60000 (96)%]     Train Loss : 0.3143

Test Loss : 0.0039      Accuracy : 96.19666666666667

Epochs : 3

Train Epoch : 3 [0/60000 (0)%]  Train Loss : 0.1508
Train Epoch : 3 [6400/60000 (11)%]      Train Loss : 0.4369
Train Epoch : 3 [12800/60000 (21)%]     Train Loss : 0.1580
Train Epoch : 3 [19200/60000 (32)%]     Train Loss : 0.1049
Train Epoch : 3 [25600/60000 (43)%]     Train Loss : 0.0789
Train Epoch : 3 [32000/60000 (53)%]     Train Loss : 0.6867
Train Epoch : 3 [38400/60000 (64)%]     Train Loss : 0.4529
Train Epoch : 3 [44800/60000 (75)%]     Train Loss : 0.3762
Train Epoch : 3 [51200/60000 (85)%]     Train Loss : 0.1195
Train Epoch : 3 [57600/60000 (96)%]     Train Loss : 0.2105

Test Loss : 0.0030      Accuracy : 97.05833333333334

Epochs : 4

Train Epoch : 4 [0/60000 (0)%]  Train Loss : 0.1001
Train Epoch : 4 [6400/60000 (11)%]      Train Loss : 0.5401
Train Epoch : 4 [12800/60000 (21)%]     Train Loss : 0.2270
Train Epoch : 4 [19200/60000 (32)%]     Train Loss : 0.0698
Train Epoch : 4 [25600/60000 (43)%]     Train Loss : 0.0377
Train Epoch : 4 [32000/60000 (53)%]     Train Loss : 0.6081
Train Epoch : 4 [38400/60000 (64)%]     Train Loss : 0.1937
Train Epoch : 4 [44800/60000 (75)%]     Train Loss : 0.1665
Train Epoch : 4 [51200/60000 (85)%]     Train Loss : 0.0801
Train Epoch : 4 [57600/60000 (96)%]     Train Loss : 0.3105

Test Loss : 0.0026      Accuracy : 97.48166666666667

Epochs : 5

Train Epoch : 5 [0/60000 (0)%]  Train Loss : 0.1048
Train Epoch : 5 [6400/60000 (11)%]      Train Loss : 0.2801
Train Epoch : 5 [12800/60000 (21)%]     Train Loss : 0.1498
Train Epoch : 5 [19200/60000 (32)%]     Train Loss : 0.0818
Train Epoch : 5 [25600/60000 (43)%]     Train Loss : 0.0878
Train Epoch : 5 [32000/60000 (53)%]     Train Loss : 0.4783
Train Epoch : 5 [38400/60000 (64)%]     Train Loss : 0.2870
Train Epoch : 5 [44800/60000 (75)%]     Train Loss : 0.0770
Train Epoch : 5 [51200/60000 (85)%]     Train Loss : 0.1258
Train Epoch : 5 [57600/60000 (96)%]     Train Loss : 0.4076

Test Loss : 0.0023      Accuracy : 97.72666666666667

...

Epochs : 30

Train Epoch : 30 [0/60000 (0)%]         Train Loss : 0.0292
Train Epoch : 30 [6400/60000 (11)%]     Train Loss : 0.2568
Train Epoch : 30 [12800/60000 (21)%]    Train Loss : 0.1643
Train Epoch : 30 [19200/60000 (32)%]    Train Loss : 0.1708
Train Epoch : 30 [25600/60000 (43)%]    Train Loss : 0.0012
Train Epoch : 30 [32000/60000 (53)%]    Train Loss : 0.0705
Train Epoch : 30 [38400/60000 (64)%]    Train Loss : 0.0481
Train Epoch : 30 [44800/60000 (75)%]    Train Loss : 0.0427
Train Epoch : 30 [51200/60000 (85)%]    Train Loss : 0.0721
Train Epoch : 30 [57600/60000 (96)%]    Train Loss : 0.0871

Test Loss : 0.0005      Accuracy : 99.47833333333334
"""
