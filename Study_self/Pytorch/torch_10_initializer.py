import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init                                    # initializer 를 사용하기 위해 torch.nn.init module 을 import 한다

import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = torchvision.datasets.MNIST(
    root = 'data',
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
    train_data, batch_size = 32
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = 32
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
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.batch_norm1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.batch_norm2(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

def weight_init(m):                                             # model 내에서 weight 를 초기화 할 부분을 설정하기 위해 함수를 정의
    if isinstance(m, nn.Linear):                                # model 을 구성하는 param 중 nn.Linear 에 해당하는 값에 대해서만 지정
        init.kaiming_uniform_(m.weight.data)                    # he_initialization 을 이용해 param 을 초기화

model = Net().to(device)
model.apply(weight_init)                                        # Net() 클래스의 인스턴스인 model 에 적용
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
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
            print('Train Epochs : {} [{}/{} ({:.0f})%] \tTrain Loss : {:.4f}'.format(
                epoch,
                batch_idx * len(image),
                len(train_loader.dataset),
                100 * batch_idx / len(train_loader),
                loss.item()
            ))

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
        accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, accuracy

for epoch in range(1, 31):
    print(f'\nEpochs : {epoch}')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = test(model, test_loader)
    print(f'\nTest Loss : {test_loss:.4f} \tAccuracy : {accuracy}')

"""
Train Epochs : 1 [0/60000 (0)%]         Train Loss : 2.9747
Train Epochs : 1 [6400/60000 (11)%]     Train Loss : 1.0148
Train Epochs : 1 [12800/60000 (21)%]    Train Loss : 1.0743
Train Epochs : 1 [19200/60000 (32)%]    Train Loss : 0.7240
Train Epochs : 1 [25600/60000 (43)%]    Train Loss : 0.4314
Train Epochs : 1 [32000/60000 (53)%]    Train Loss : 0.5979
Train Epochs : 1 [38400/60000 (64)%]    Train Loss : 0.4233
Train Epochs : 1 [44800/60000 (75)%]    Train Loss : 0.5413
Train Epochs : 1 [51200/60000 (85)%]    Train Loss : 0.3637
Train Epochs : 1 [57600/60000 (96)%]    Train Loss : 0.5416

Test Loss : 0.0078      Accuracy : 92.52

Epochs : 2
Train Epochs : 2 [0/60000 (0)%]         Train Loss : 0.4144
Train Epochs : 2 [6400/60000 (11)%]     Train Loss : 0.5667
Train Epochs : 2 [12800/60000 (21)%]    Train Loss : 0.4351
Train Epochs : 2 [19200/60000 (32)%]    Train Loss : 0.5256
Train Epochs : 2 [25600/60000 (43)%]    Train Loss : 0.2696
Train Epochs : 2 [32000/60000 (53)%]    Train Loss : 0.6484
Train Epochs : 2 [38400/60000 (64)%]    Train Loss : 0.4887
Train Epochs : 2 [44800/60000 (75)%]    Train Loss : 0.4514
Train Epochs : 2 [51200/60000 (85)%]    Train Loss : 0.2242
Train Epochs : 2 [57600/60000 (96)%]    Train Loss : 0.5886

Test Loss : 0.0061      Accuracy : 94.11

Epochs : 3
Train Epochs : 3 [0/60000 (0)%]         Train Loss : 0.2667
Train Epochs : 3 [6400/60000 (11)%]     Train Loss : 0.7228
Train Epochs : 3 [12800/60000 (21)%]    Train Loss : 0.3904
Train Epochs : 3 [19200/60000 (32)%]    Train Loss : 0.1711
Train Epochs : 3 [25600/60000 (43)%]    Train Loss : 0.1788
Train Epochs : 3 [32000/60000 (53)%]    Train Loss : 0.4055
Train Epochs : 3 [38400/60000 (64)%]    Train Loss : 0.6374
Train Epochs : 3 [44800/60000 (75)%]    Train Loss : 0.5396
Train Epochs : 3 [51200/60000 (85)%]    Train Loss : 0.1688
Train Epochs : 3 [57600/60000 (96)%]    Train Loss : 0.4227

Test Loss : 0.0053      Accuracy : 94.94

Epochs : 4
Train Epochs : 4 [0/60000 (0)%]         Train Loss : 0.2344
Train Epochs : 4 [6400/60000 (11)%]     Train Loss : 0.4949
Train Epochs : 4 [12800/60000 (21)%]    Train Loss : 0.3881
Train Epochs : 4 [19200/60000 (32)%]    Train Loss : 0.1857
Train Epochs : 4 [25600/60000 (43)%]    Train Loss : 0.2784
Train Epochs : 4 [32000/60000 (53)%]    Train Loss : 0.3942
Train Epochs : 4 [38400/60000 (64)%]    Train Loss : 0.3373
Train Epochs : 4 [44800/60000 (75)%]    Train Loss : 0.3892
Train Epochs : 4 [51200/60000 (85)%]    Train Loss : 0.1401
Train Epochs : 4 [57600/60000 (96)%]    Train Loss : 0.4615

Test Loss : 0.0049      Accuracy : 95.08

Epochs : 5
Train Epochs : 5 [0/60000 (0)%]         Train Loss : 0.1550
Train Epochs : 5 [6400/60000 (11)%]     Train Loss : 0.6761
Train Epochs : 5 [12800/60000 (21)%]    Train Loss : 0.1680
Train Epochs : 5 [19200/60000 (32)%]    Train Loss : 0.0789
Train Epochs : 5 [25600/60000 (43)%]    Train Loss : 0.1390
Train Epochs : 5 [32000/60000 (53)%]    Train Loss : 0.5535
Train Epochs : 5 [38400/60000 (64)%]    Train Loss : 0.3349
Train Epochs : 5 [44800/60000 (75)%]    Train Loss : 0.4343
Train Epochs : 5 [51200/60000 (85)%]    Train Loss : 0.2813
Train Epochs : 5 [57600/60000 (96)%]    Train Loss : 0.6243

Test Loss : 0.0046      Accuracy : 95.55

...

Epochs : 30
Train Epochs : 30 [0/60000 (0)%]        Train Loss : 0.2312
Train Epochs : 30 [6400/60000 (11)%]    Train Loss : 0.4320
Train Epochs : 30 [12800/60000 (21)%]   Train Loss : 0.2027
Train Epochs : 30 [19200/60000 (32)%]   Train Loss : 0.1016
Train Epochs : 30 [25600/60000 (43)%]   Train Loss : 0.0955
Train Epochs : 30 [32000/60000 (53)%]   Train Loss : 0.4507
Train Epochs : 30 [38400/60000 (64)%]   Train Loss : 0.3351
Train Epochs : 30 [44800/60000 (75)%]   Train Loss : 0.3093
Train Epochs : 30 [51200/60000 (85)%]   Train Loss : 0.1155
Train Epochs : 30 [57600/60000 (96)%]   Train Loss : 0.3996

Test Loss : 0.0037      Accuracy : 96.77
"""