import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')
# current device is cuda

batch_size = 32
epochs = 10

train_data = datasets.CIFAR10(
    root = 'data',
    download=False,
    train=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size
)

for (x_train, y_train) in train_loader:
    print('x_train shape : ', x_train.size(), 'x_train type : ', x_train.type())
    print('y_train shape : ', y_train.size(), 'y_train type : ', y_train.type())
    break
# x_train shape :  torch.Size([32, 3, 32, 32]) x_train type :  torch.FloatTensor
# y_train shape :  torch.Size([32]) y_train type :  torch.LongTensor

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(np.transpose(x_train[i], (1, 2, 0)))
    plt.title('class : ' + str(y_train[i].item()))
plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-3
)

print(model)
# Net(
#   (fc1): Linear(in_features=3072, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )

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
            print(f'Epochs : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f})% Train Loss : {loss.item():.4f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = output.max(1, keepdim = True)[1]
            test_loss += criterion(output, label).item()
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, accuracy

for epoch in range(1, epochs + 1):
    print(f'Epochs : {epoch}\n')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = test(model, test_loader)
    print(f'\nTest Loss : {test_loss:.4f} \t Accuracy : {accuracy} %\n')

"""
Epochs : 1

Epochs : 1 [0/50000 (0)% Train Loss : 2.3065
Epochs : 1 [6400/50000 (13)% Train Loss : 1.8117
Epochs : 1 [12800/50000 (26)% Train Loss : 1.6493
Epochs : 1 [19200/50000 (38)% Train Loss : 1.9561
Epochs : 1 [25600/50000 (51)% Train Loss : 1.7607
Epochs : 1 [32000/50000 (64)% Train Loss : 1.7641
Epochs : 1 [38400/50000 (77)% Train Loss : 1.8228
Epochs : 1 [44800/50000 (90)% Train Loss : 1.7119

Test Loss : 0.0532       Accuracy : 38.99 %

Epochs : 2

Epochs : 2 [0/50000 (0)% Train Loss : 1.8231
Epochs : 2 [6400/50000 (13)% Train Loss : 1.4989
Epochs : 2 [12800/50000 (26)% Train Loss : 1.3404
Epochs : 2 [19200/50000 (38)% Train Loss : 1.6641
Epochs : 2 [25600/50000 (51)% Train Loss : 1.5947
Epochs : 2 [32000/50000 (64)% Train Loss : 1.6490
Epochs : 2 [38400/50000 (77)% Train Loss : 1.6002
Epochs : 2 [44800/50000 (90)% Train Loss : 1.6281

Test Loss : 0.0508       Accuracy : 42.154 %

Epochs : 3

Epochs : 3 [0/50000 (0)% Train Loss : 1.8265
Epochs : 3 [6400/50000 (13)% Train Loss : 1.3791
Epochs : 3 [12800/50000 (26)% Train Loss : 1.3333
Epochs : 3 [19200/50000 (38)% Train Loss : 1.6325
Epochs : 3 [25600/50000 (51)% Train Loss : 1.5607
Epochs : 3 [32000/50000 (64)% Train Loss : 1.5982
Epochs : 3 [38400/50000 (77)% Train Loss : 1.5383
Epochs : 3 [44800/50000 (90)% Train Loss : 1.5662

Test Loss : 0.0483       Accuracy : 44.732 %

Epochs : 4

Epochs : 4 [0/50000 (0)% Train Loss : 1.7258
Epochs : 4 [6400/50000 (13)% Train Loss : 1.3582
Epochs : 4 [12800/50000 (26)% Train Loss : 1.2796
Epochs : 4 [19200/50000 (38)% Train Loss : 1.6315
Epochs : 4 [25600/50000 (51)% Train Loss : 1.4517
Epochs : 4 [32000/50000 (64)% Train Loss : 1.5283
Epochs : 4 [38400/50000 (77)% Train Loss : 1.4833
Epochs : 4 [44800/50000 (90)% Train Loss : 1.5138

Test Loss : 0.0467       Accuracy : 46.686 %

Epochs : 5

Epochs : 5 [0/50000 (0)% Train Loss : 1.6043
Epochs : 5 [6400/50000 (13)% Train Loss : 1.3056
Epochs : 5 [12800/50000 (26)% Train Loss : 1.2734
Epochs : 5 [19200/50000 (38)% Train Loss : 1.5514
Epochs : 5 [25600/50000 (51)% Train Loss : 1.3476
Epochs : 5 [32000/50000 (64)% Train Loss : 1.4622
Epochs : 5 [38400/50000 (77)% Train Loss : 1.4365
Epochs : 5 [44800/50000 (90)% Train Loss : 1.5356

Test Loss : 0.0454       Accuracy : 47.93 %

Epochs : 6

Epochs : 6 [0/50000 (0)% Train Loss : 1.5894
Epochs : 6 [6400/50000 (13)% Train Loss : 1.3724
Epochs : 6 [12800/50000 (26)% Train Loss : 1.2464
Epochs : 6 [19200/50000 (38)% Train Loss : 1.5307
Epochs : 6 [25600/50000 (51)% Train Loss : 1.3873
Epochs : 6 [32000/50000 (64)% Train Loss : 1.3981
Epochs : 6 [38400/50000 (77)% Train Loss : 1.3690
Epochs : 6 [44800/50000 (90)% Train Loss : 1.5522

Test Loss : 0.0458       Accuracy : 47.564 %

Epochs : 7

Epochs : 7 [0/50000 (0)% Train Loss : 1.5084
Epochs : 7 [6400/50000 (13)% Train Loss : 1.2976
Epochs : 7 [12800/50000 (26)% Train Loss : 1.3177
Epochs : 7 [19200/50000 (38)% Train Loss : 1.4879
Epochs : 7 [25600/50000 (51)% Train Loss : 1.2517
Epochs : 7 [32000/50000 (64)% Train Loss : 1.4049
Epochs : 7 [38400/50000 (77)% Train Loss : 1.3508
Epochs : 7 [44800/50000 (90)% Train Loss : 1.4669

Test Loss : 0.0442       Accuracy : 49.332 %

Epochs : 8

Epochs : 8 [0/50000 (0)% Train Loss : 1.4529
Epochs : 8 [6400/50000 (13)% Train Loss : 1.3757
Epochs : 8 [12800/50000 (26)% Train Loss : 1.3254
Epochs : 8 [19200/50000 (38)% Train Loss : 1.4631
Epochs : 8 [25600/50000 (51)% Train Loss : 1.2700
Epochs : 8 [32000/50000 (64)% Train Loss : 1.3879
Epochs : 8 [38400/50000 (77)% Train Loss : 1.4073
Epochs : 8 [44800/50000 (90)% Train Loss : 1.4556

Test Loss : 0.0433       Accuracy : 50.358 %

Epochs : 9

Epochs : 9 [0/50000 (0)% Train Loss : 1.4977
Epochs : 9 [6400/50000 (13)% Train Loss : 1.2704
Epochs : 9 [12800/50000 (26)% Train Loss : 1.3013
Epochs : 9 [19200/50000 (38)% Train Loss : 1.4445
Epochs : 9 [25600/50000 (51)% Train Loss : 1.3433
Epochs : 9 [32000/50000 (64)% Train Loss : 1.3628
Epochs : 9 [38400/50000 (77)% Train Loss : 1.3693
Epochs : 9 [44800/50000 (90)% Train Loss : 1.5154

Test Loss : 0.0431       Accuracy : 50.56 %

Epochs : 10

Epochs : 10 [0/50000 (0)% Train Loss : 1.4353
Epochs : 10 [6400/50000 (13)% Train Loss : 1.3915
Epochs : 10 [12800/50000 (26)% Train Loss : 1.2616
Epochs : 10 [19200/50000 (38)% Train Loss : 1.4519
Epochs : 10 [25600/50000 (51)% Train Loss : 1.2802
Epochs : 10 [32000/50000 (64)% Train Loss : 1.3062
Epochs : 10 [38400/50000 (77)% Train Loss : 1.2741
Epochs : 10 [44800/50000 (90)% Train Loss : 1.4836

Test Loss : 0.0424       Accuracy : 51.688 %
"""