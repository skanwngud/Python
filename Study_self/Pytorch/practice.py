# module import
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn                           # 딥러닝에 관련 된 모듈
import torch.nn.functional as F                 # 위의 모듈 중에서도 자주 사용 되는 함수
import os

from torchvision import transforms, datasets

os.environ['KMP_DUPLICATE_LIB_OK']='True'       # OMP error 관련

# equipment check
if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')
print('current device : {}'.format(device))     # cpu

batch_size = 32                                 # mini_batch 1개 당 32개로 구성 되어있음
epochs = 10                                     # 훈련 횟수

# data download (in this case : MNIST)
train_dataset = datasets.MNIST(
    root='data',                                # 상위 directory 의 data 에 데이터 저장
    train=True,                                 # data 를 train_set 으로 분리해서 사용
    download=True,                              # download 를 시행할 것인지
    transform=transforms.ToTensor(),            # ToTensor() 를 사용하면 기본적인 전처리 (픽셀값을 0~1로 정규화) 실시
)

test_dataset = datasets.MNIST(
    root= 'data',
    train=False,                                # data 를 test_set 으로 분리해서 사용
    download=True,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(     # mini_batch 단위로 분리해서 지정
    dataset=train_dataset,                      # dataset 지정
    batch_size=batch_size,                      # mini_batch 1개 단위를 구성하는 데이터의 갯수를 지정
    shuffle=True,                               # dataset 을 random 하게 섞어 과적합을 방지
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# check data
for (x_train, y_train) in train_loader:
    print('x_train : ', x_train.size(), 'type : ', x_train.type())
    print('y_train : ', y_train.size(), 'type : ', y_train.type())
    break

# x_train :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor
# 32개의 이미지 데이터가 1개의 mini_batch size 로 묶여있고, (28, 28, 1) 의 이미지 데이터
# FloatTensor 의 형태

# y_train :  torch.Size([32]) type :  torch.LongTensor
# 32개의 이미지 데이터가 하나의 label 을 갖기 때문에 (32,) 의 shape
# LongTensor 의 형태 (숫자를 표현할 때 쓰는 data type (e.g. short, long, longlong....))

# visualize data
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize * 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(x_train[i, :, :, :].numpy().reshape(28, 28), cmap = 'gray_r')
    plt.title('Class : ' + str(y_train[i].item()))
plt.show()

# MLP(Multi Layer Perceptron) modeling
class Net(nn.Module):                           # pytorch module 내에 딥러닝 모델 관련 기본 함수를 포함하는 nn.Module 클래스를 상속받는 Net 클래스를 정의
    def __init__(self):                         # Net 클래스의 인스턴스를 생성했을 때 지니게 되는 성질 (즉, 인스턴스가 생성 되었을 때 무조건 실행 되는 부분)
        super(Net, self).__init__()             # nn.Module 내에 있는 메서드를 상속받아 이용
        self.fc1 = nn.Linear(28 * 28, 512)      # 첫 번째 Fully Connected Layer 정의 (input layer)
        self.fc2 = nn.Linear(512, 256)          # 두 번째 Fully Connected Layer 정의 (hidden layer)
        self.fc3 = nn.Linear(256, 10)           # 마지막 Fully Connected Layer 정의 (output layer)
    
    def forward(self, x):                       # forward propagation 정의
        x = x.view(-1, 28 * 28)                 # view 메서드를 이용하여 input shape 를 layer 에 맞춰줌 (Flatten)
        x = self.fc1(x)                         # 첫 번째 Fully Connected Layer 를 통과 시킴
        x = F.sigmoid(x)                        # fc1 를 통과해 나온 데이터를 sigmoid 함수로 계산
        x = self.fc2(x)                         # 첫 번째 layer 의 반복
        x = F.sigmoid(x)
        x = self.fc3(x)                         # 첫 번째 layer 의 반복
        x = F.log_softmax(x, dim = 1)           # fc3 를 통과해 나온 데이터를 log_softmax 함수로 계산시켜 predict 값을 여러가지 경우의 수에서 가장 높은 확률 값을 가진 수를 return 함
                                                # softmax 가 아니라 log_softmax 인 이유는 back propagation 을 진행할 때 loss 계산을 좀 더 원활히 하기 위함
        return x                                # 최종값인 x 를 return 시킴

# define optimizer, objective function
model = Net().to(device)                        # 위에서 정의한 model 을 device 에 할당 (cuda 를 사용할 경우 cuda 를 쓰기 위함)
optimizer = torch.optim.SGD(                    # back propagation 을 진행할 때 사용하는 optimizer 를 정의
    model.parameters(),                         # model 내에 있는 parameters 를 업데이트 시킴
    lr = 1e-2, momentum = 0.5)                  # learning rate 와 momentum (관성) 을 정의
criterion = nn.CrossEntropyLoss()               # loss 정의
print(model)

# Net(
#   (fc1): Linear(in_features=784, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )

# model evaluate
def train(model, train_loader, optimzer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimzer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch : {} [{}/{}({:.0f}%)]\tTrain Loss : {:.6f}'.format(
                epochs,
                batch_idx * len(image),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))