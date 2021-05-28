# import module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision

batch_size = 32
epochs = 30

if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')
print('current device is {}.'.format(device))


# data
train_data = torchvision.datasets.MNIST(
    root = 'data',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.MNIST(
    root='data',
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, 
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
)

# mlp
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5                                         # 몇 퍼센트의 노드에 대해 가중치를 계산하지 않을 것인가, 0.5 = 50%

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)   # sigmoid 결괏값에 대해 dropout 을 적용하는 부분
                                                                        # 훈련 과정 중에만 dropout 이 적용 돼야하기 때문에
                                                                        # model.train() 에서는 training = True, model.eval() 에서는 training = False 이다
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)

        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum=0.5)
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
            print(
                '[Train Epoch : {} [{}/{}({:.0f})%] \tTrain Loss : {:.6f}'.format(
                    epoch,
                    batch_idx * len(image),
                    len(train_loader.dataset),
                    100 * batch_idx/ len(train_loader),
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
    test_correct = 100. * correct / len(test_loader.dataset)
    return test_loss, test_correct

for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer ,log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print('\nEpochs : {} \t Test_loss : {} \t Test_accuracy : {}\n'.format(
        epoch,
        test_loss,
        test_accuracy
    ))

"""
[Train Epoch : 1 [0/60000(0)%]  Train Loss : 2.274121
[Train Epoch : 1 [6400/60000(11)%]      Train Loss : 2.272378
[Train Epoch : 1 [12800/60000(21)%]     Train Loss : 2.220892
[Train Epoch : 1 [19200/60000(32)%]     Train Loss : 2.354084
[Train Epoch : 1 [25600/60000(43)%]     Train Loss : 2.352526
[Train Epoch : 1 [32000/60000(53)%]     Train Loss : 2.370736
[Train Epoch : 1 [38400/60000(64)%]     Train Loss : 2.350482
[Train Epoch : 1 [44800/60000(75)%]     Train Loss : 2.309633
[Train Epoch : 1 [51200/60000(85)%]     Train Loss : 2.352006
[Train Epoch : 1 [57600/60000(96)%]     Train Loss : 2.310157

Epochs : 1       Test_loss : 0.07111180396080018         Test_accuracy : 33.05

[Train Epoch : 2 [0/60000(0)%]  Train Loss : 2.294412
[Train Epoch : 2 [6400/60000(11)%]      Train Loss : 2.265640
[Train Epoch : 2 [12800/60000(21)%]     Train Loss : 2.223791
[Train Epoch : 2 [19200/60000(32)%]     Train Loss : 2.323995
[Train Epoch : 2 [25600/60000(43)%]     Train Loss : 2.243240
[Train Epoch : 2 [32000/60000(53)%]     Train Loss : 2.252417
[Train Epoch : 2 [38400/60000(64)%]     Train Loss : 2.129033
[Train Epoch : 2 [44800/60000(75)%]     Train Loss : 2.289210
[Train Epoch : 2 [51200/60000(85)%]     Train Loss : 2.127997
[Train Epoch : 2 [57600/60000(96)%]     Train Loss : 2.051547

Epochs : 2       Test_loss : 0.062180495238304136        Test_accuracy : 47.48

[Train Epoch : 3 [0/60000(0)%]  Train Loss : 1.963862
[Train Epoch : 3 [6400/60000(11)%]      Train Loss : 1.826844
[Train Epoch : 3 [12800/60000(21)%]     Train Loss : 1.912353
[Train Epoch : 3 [19200/60000(32)%]     Train Loss : 1.761548
[Train Epoch : 3 [25600/60000(43)%]     Train Loss : 1.567480
[Train Epoch : 3 [32000/60000(53)%]     Train Loss : 1.672499
[Train Epoch : 3 [38400/60000(64)%]     Train Loss : 1.443137
[Train Epoch : 3 [44800/60000(75)%]     Train Loss : 1.473758
[Train Epoch : 3 [51200/60000(85)%]     Train Loss : 1.351370
[Train Epoch : 3 [57600/60000(96)%]     Train Loss : 1.108904

Epochs : 3       Test_loss : 0.03569663505554199         Test_accuracy : 64.57

[Train Epoch : 4 [0/60000(0)%]  Train Loss : 1.378917
[Train Epoch : 4 [6400/60000(11)%]      Train Loss : 1.181152
[Train Epoch : 4 [12800/60000(21)%]     Train Loss : 1.335566
[Train Epoch : 4 [19200/60000(32)%]     Train Loss : 1.096408
[Train Epoch : 4 [25600/60000(43)%]     Train Loss : 1.029480
[Train Epoch : 4 [32000/60000(53)%]     Train Loss : 0.922285
[Train Epoch : 4 [38400/60000(64)%]     Train Loss : 1.022542
[Train Epoch : 4 [44800/60000(75)%]     Train Loss : 1.075613
[Train Epoch : 4 [51200/60000(85)%]     Train Loss : 1.129820
[Train Epoch : 4 [57600/60000(96)%]     Train Loss : 1.048444

Epochs : 4       Test_loss : 0.02691202079951763         Test_accuracy : 71.65

[Train Epoch : 5 [0/60000(0)%]  Train Loss : 1.066700
[Train Epoch : 5 [6400/60000(11)%]      Train Loss : 1.105929
[Train Epoch : 5 [12800/60000(21)%]     Train Loss : 1.058742
[Train Epoch : 5 [19200/60000(32)%]     Train Loss : 0.844939
[Train Epoch : 5 [25600/60000(43)%]     Train Loss : 0.916778
[Train Epoch : 5 [32000/60000(53)%]     Train Loss : 0.863014
[Train Epoch : 5 [38400/60000(64)%]     Train Loss : 1.181810
[Train Epoch : 5 [44800/60000(75)%]     Train Loss : 0.919009
[Train Epoch : 5 [51200/60000(85)%]     Train Loss : 0.939848
[Train Epoch : 5 [57600/60000(96)%]     Train Loss : 0.973305

Epochs : 5       Test_loss : 0.02335604763031006         Test_accuracy : 76.33

...

[Train Epoch : 30 [0/60000(0)%]         Train Loss : 0.318306
[Train Epoch : 30 [6400/60000(11)%]     Train Loss : 0.434127
[Train Epoch : 30 [12800/60000(21)%]    Train Loss : 0.272882
[Train Epoch : 30 [19200/60000(32)%]    Train Loss : 0.279770
[Train Epoch : 30 [25600/60000(43)%]    Train Loss : 0.256902
[Train Epoch : 30 [32000/60000(53)%]    Train Loss : 0.446335
[Train Epoch : 30 [38400/60000(64)%]    Train Loss : 0.209806
[Train Epoch : 30 [44800/60000(75)%]    Train Loss : 0.443696
[Train Epoch : 30 [51200/60000(85)%]    Train Loss : 0.509579
[Train Epoch : 30 [57600/60000(96)%]    Train Loss : 0.284951

Epochs : 30      Test_loss : 0.008348281732667237        Test_accuracy : 91.97
"""