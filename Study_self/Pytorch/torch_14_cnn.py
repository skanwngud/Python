import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')

epochs = 10

train_data = datasets.CIFAR10(
    root='data',
    download=False,
    train=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.CIFAR10(
    root='data',
    download=False,
    train=False,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(                             # 이미지 데이터를 convolution 연산하는 filter 를 정의
            in_channels=3,                                  # 인풋 채널수와 이미지 데이터의 채널수를 맞춰야한다, r, g, b 의 픽셀에 대해 동시에 convolution 연산을 해야하기 위함
            out_channels=8,                                 # convolution 연산을 하고 몇 겹의 feature map 을 쌓을지에 대한 채널. 8겹의 채널이 쌓인다
            kernel_size=3,                                  # 스칼라값은 가로 * 세로의 값이므로 3 은 3 * 3 이 된다
                                                            # 해당 크기의 filter 가 이미지를 훑으며 연산함
            padding=1                                       # 이미지 데이터의 외측에 0 으로 채우면서 모든 픽셀에 연산이 동일하게 진행 되게끔한다
                                                            # 1을 지정한 경우에는 외측에서 1 칸을 0으로 채우겠다는 의미
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,                                  # conv1 layer 를 통해 나온 아웃풋 데이터의 채널과 일치 시켜줘야한다
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(                           # convolution 연산을 통해 생성 된 feature map 을 전부 이용하는 것이 아닌 부분적으로 이용하기 위해 사용
                                                            # MaxPool은 feature map 에서 지정 된 크기 내 가장 큰 값만 이용
            kernel_size=2,                                  # feature map 내에 크기 2 * 2 크기의 filter 가 돌아다니면서 pooling 을 진행한다
                                                            # maxpool 이기 때문에 생성 된 4개의 값중 3개의 값은 버려지고 가장 큰값만 취한다
            stride=2                                        # filter 가 feature map 을 움직일 때 몇 칸씩 이동할지
        )

        self.fc1 = nn.Linear(8 * 8* 16, 64)                 # 이미 위에서 해당 픽셀들의 특징들을 전부 뽑았으므로 1차원으로 늘려도 (flatten) 문제가 없다
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 8*8*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
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
            print(f'Epochs : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f})% Train Loss : {loss.item():.4f}]')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = output.max(1, keepdim=True)[1]
            test_loss += criterion(output, label).item()
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, accuracy

for epoch in range(1, epochs + 1):
    print(f'Epochs : {epoch}\n')
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = test(model, test_loader)
    print(f'\nTest Loss : {test_loss} \tAccuracy : {accuracy}\n')

"""
Epochs : 1

Epochs : 1 [0/50000 (0)% Train Loss : 2.2868]
Epochs : 1 [6400/50000 (13)% Train Loss : 1.8427]
Epochs : 1 [12800/50000 (26)% Train Loss : 1.8330]
Epochs : 1 [19200/50000 (38)% Train Loss : 2.0226]
Epochs : 1 [25600/50000 (51)% Train Loss : 1.6555]
Epochs : 1 [32000/50000 (64)% Train Loss : 1.7476]
Epochs : 1 [38400/50000 (77)% Train Loss : 1.7286]
Epochs : 1 [44800/50000 (90)% Train Loss : 1.4814]

Test Loss : 0.049166537988185884        Accuracy : 42.39

Epochs : 2

Epochs : 2 [0/50000 (0)% Train Loss : 1.6195]
Epochs : 2 [6400/50000 (13)% Train Loss : 1.2935]
Epochs : 2 [12800/50000 (26)% Train Loss : 1.3461]
Epochs : 2 [19200/50000 (38)% Train Loss : 1.6242]
Epochs : 2 [25600/50000 (51)% Train Loss : 1.2795]
Epochs : 2 [32000/50000 (64)% Train Loss : 1.4791]
Epochs : 2 [38400/50000 (77)% Train Loss : 1.5102]
Epochs : 2 [44800/50000 (90)% Train Loss : 1.4145]

Test Loss : 0.04581127371191979         Accuracy : 46.62

Epochs : 3

Epochs : 3 [0/50000 (0)% Train Loss : 1.4838]
Epochs : 3 [6400/50000 (13)% Train Loss : 1.2601]
Epochs : 3 [12800/50000 (26)% Train Loss : 1.0489]
Epochs : 3 [19200/50000 (38)% Train Loss : 1.5277]
Epochs : 3 [25600/50000 (51)% Train Loss : 1.0721]
Epochs : 3 [32000/50000 (64)% Train Loss : 1.3127]
Epochs : 3 [38400/50000 (77)% Train Loss : 1.2737]
Epochs : 3 [44800/50000 (90)% Train Loss : 1.3712]

Test Loss : 0.04260787582397461         Accuracy : 50.8

Epochs : 4

Epochs : 4 [0/50000 (0)% Train Loss : 1.3870]
Epochs : 4 [6400/50000 (13)% Train Loss : 1.1811]
Epochs : 4 [12800/50000 (26)% Train Loss : 0.9553]
Epochs : 4 [19200/50000 (38)% Train Loss : 1.4837]
Epochs : 4 [25600/50000 (51)% Train Loss : 1.0222]
Epochs : 4 [32000/50000 (64)% Train Loss : 1.2544]
Epochs : 4 [38400/50000 (77)% Train Loss : 1.1254]
Epochs : 4 [44800/50000 (90)% Train Loss : 1.3262]

Test Loss : 0.0390545175909996  Accuracy : 55.5

Epochs : 5

Epochs : 5 [0/50000 (0)% Train Loss : 1.2907]
Epochs : 5 [6400/50000 (13)% Train Loss : 1.1041]
Epochs : 5 [12800/50000 (26)% Train Loss : 0.9877]
Epochs : 5 [19200/50000 (38)% Train Loss : 1.3768]
Epochs : 5 [25600/50000 (51)% Train Loss : 1.0175]
Epochs : 5 [32000/50000 (64)% Train Loss : 1.1730]
Epochs : 5 [38400/50000 (77)% Train Loss : 1.0617]
Epochs : 5 [44800/50000 (90)% Train Loss : 1.2552]

Test Loss : 0.03698785253167152         Accuracy : 57.95

...

Epochs : 10

Epochs : 10 [0/50000 (0)% Train Loss : 0.9783]
Epochs : 10 [6400/50000 (13)% Train Loss : 1.0823]
Epochs : 10 [12800/50000 (26)% Train Loss : 1.0487]
Epochs : 10 [19200/50000 (38)% Train Loss : 1.2592]
Epochs : 10 [25600/50000 (51)% Train Loss : 1.0003]
Epochs : 10 [32000/50000 (64)% Train Loss : 1.0343]
Epochs : 10 [38400/50000 (77)% Train Loss : 0.9329]
Epochs : 10 [44800/50000 (90)% Train Loss : 1.0446]

Test Loss : 0.03460100528001785         Accuracy : 61.68
"""