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
def train(model, train_loader, optimizer, log_interval):
    model.train()                               # MLP 모델을 정의하고 훈련시킴
    for batch_idx, (image, label) in enumerate(train_loader): # 한 배치당 image 와 label 을 묶어서 훈련시킴
        image = image.to(device)                # mini_batch 에 있는 image 를 학습시키기 위해 기존에 정의 된 장비에 할당
        label = label.to(device)                # mini_batch 에 있는 label 을 학습시키기 위해 기존에 정의 된 장비에 할당
        optimizer.zero_grad()                    # optimizer 를 통해 앞서 계산 된 gradient 를 초기화
        output = model(image)                   # 장비에 할당 된 image 를 model 의 input 으로 이용해 output 을 계산
        loss = criterion(output, label)         # output 을 cross entropy loss 를 이용해 loss 값을 계산
        loss.backward()                         # 계산 된 loss 값을 back propagation 시킴
        optimizer.step()                        # 각 파라미터에 할당 된 gradient 를 통해 업데이트 시킴

        if batch_idx % log_interval == 0:
            print('Train Epoch : {} [{}/{}({:.0f}%)]\tTrain Loss : {:.6f}'.format(
                epoch,
                batch_idx * len(image),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

def evalutate(model, test_loader):
    model.eval()                                # 학습 과정 혹은 학습이 완료 된 모델을 학습 상태가 아닌 평가 상태로 지정
    test_loss = 0                               # test_loader 내의 데이터를 통해 loss 값을 계산하기 위해 0 으로 지정
    correct = 0                                 # model 이 올바른 class 로 분류할 경우를 세기 위해 임시로 0 으로 지정

    with torch.no_grad():                       # 평가 과정에서는 gradient 가 업데이트 되면 안 되기 때문에 해당 module 안에서 평가함
        for image, label in test_loader:        # image 와 label 이 mini_batch 단위로 저장이 되어있기 때문에 반복문을 이용해 차례대로 접근
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()                        # CrossEntropyLoss 를 통해 나온 결과값을 test_loss 에 업데이트
            prediction = output.max(1, keepdim = True)[1]                       # 계산 된 벡터값 중 가장 큰 값인 위치에 대해 해당 위치에 대응하는 클래스로 예측했다고 판단
            correct += prediction.eq(label.view_as(prediction)).sum().item()    # 예측한 결과값과 실제 결과값이 일치한 횟수를 저장

    test_loss /= len(test_loader.dataset)                                       # 업데이트 된 전체 loss 값을 test_loader 내에 존재하는 mini_batch 데이터 갯수만큼 나눔
    test_accuracy = 100. * correct / len(test_loader.dataset)                   # test_loader 데이터로 얼마나 맞췄는지를 계산
    return test_loss, test_accuracy                                             # 최종적으로 loss 값과 accuracy 값을 return

for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, log_interval=200)                     # log_interval == iterations / mini_batch 가 200번 끝날 때마다 출력
    test_loss, test_accuracy = evalutate(model, test_loader)                    # 각 epoch 별로 출력 되는 loss 와 accuracy 값을 계산
    print('\n[Epoch : {}], \tTest Loss : {:.4f}, \tTest Accuracy : {:.2f}%\n'.format(epoch, test_loss, test_accuracy))

"""
Train Epoch : 1 [0/60000(0%)]   Train Loss : 2.276873
Train Epoch : 1 [6400/60000(11%)]       Train Loss : 2.307976
Train Epoch : 1 [12800/60000(21%)]      Train Loss : 2.259264
Train Epoch : 1 [19200/60000(32%)]      Train Loss : 2.279662
Train Epoch : 1 [25600/60000(43%)]      Train Loss : 2.335363
Train Epoch : 1 [32000/60000(53%)]      Train Loss : 2.305229
Train Epoch : 1 [38400/60000(64%)]      Train Loss : 2.283150
Train Epoch : 1 [44800/60000(75%)]      Train Loss : 2.288420
Train Epoch : 1 [51200/60000(85%)]      Train Loss : 2.256940
Train Epoch : 1 [57600/60000(96%)]      Train Loss : 2.229723

[Epoch : 1],    Test Loss : 0.0699,     Test Accuracy : 27.28%

Train Epoch : 2 [0/60000(0%)]   Train Loss : 2.181584
Train Epoch : 2 [6400/60000(11%)]       Train Loss : 2.224923
Train Epoch : 2 [12800/60000(21%)]      Train Loss : 2.125711
Train Epoch : 2 [19200/60000(32%)]      Train Loss : 2.068018
Train Epoch : 2 [25600/60000(43%)]      Train Loss : 1.882562
Train Epoch : 2 [32000/60000(53%)]      Train Loss : 1.837357
Train Epoch : 2 [38400/60000(64%)]      Train Loss : 1.689243
Train Epoch : 2 [44800/60000(75%)]      Train Loss : 1.642618
Train Epoch : 2 [51200/60000(85%)]      Train Loss : 1.445550
Train Epoch : 2 [57600/60000(96%)]      Train Loss : 1.343673

[Epoch : 2],    Test Loss : 0.0382,     Test Accuracy : 60.59%

Train Epoch : 3 [0/60000(0%)]   Train Loss : 1.348246
Train Epoch : 3 [6400/60000(11%)]       Train Loss : 1.176393
Train Epoch : 3 [12800/60000(21%)]      Train Loss : 1.029263
Train Epoch : 3 [19200/60000(32%)]      Train Loss : 0.800979
Train Epoch : 3 [25600/60000(43%)]      Train Loss : 0.752938
Train Epoch : 3 [32000/60000(53%)]      Train Loss : 1.137844
Train Epoch : 3 [38400/60000(64%)]      Train Loss : 0.846454
Train Epoch : 3 [44800/60000(75%)]      Train Loss : 0.740875
Train Epoch : 3 [51200/60000(85%)]      Train Loss : 0.696743
Train Epoch : 3 [57600/60000(96%)]      Train Loss : 0.665524

[Epoch : 3],    Test Loss : 0.0235,     Test Accuracy : 76.53%

Train Epoch : 4 [0/60000(0%)]   Train Loss : 0.730363
Train Epoch : 4 [6400/60000(11%)]       Train Loss : 0.630623
Train Epoch : 4 [12800/60000(21%)]      Train Loss : 0.733060
Train Epoch : 4 [19200/60000(32%)]      Train Loss : 0.546238
Train Epoch : 4 [25600/60000(43%)]      Train Loss : 0.576465
Train Epoch : 4 [32000/60000(53%)]      Train Loss : 0.678074
Train Epoch : 4 [38400/60000(64%)]      Train Loss : 0.542645
Train Epoch : 4 [44800/60000(75%)]      Train Loss : 0.774299
Train Epoch : 4 [51200/60000(85%)]      Train Loss : 0.857701
Train Epoch : 4 [57600/60000(96%)]      Train Loss : 0.356105

[Epoch : 4],    Test Loss : 0.0178,     Test Accuracy : 83.42%

Train Epoch : 5 [0/60000(0%)]   Train Loss : 0.576185
Train Epoch : 5 [6400/60000(11%)]       Train Loss : 0.439005
Train Epoch : 5 [12800/60000(21%)]      Train Loss : 0.424895
Train Epoch : 5 [19200/60000(32%)]      Train Loss : 0.536287
Train Epoch : 5 [25600/60000(43%)]      Train Loss : 0.543942
Train Epoch : 5 [32000/60000(53%)]      Train Loss : 0.362133
Train Epoch : 5 [38400/60000(64%)]      Train Loss : 0.337298
Train Epoch : 5 [44800/60000(75%)]      Train Loss : 0.546676
Train Epoch : 5 [51200/60000(85%)]      Train Loss : 0.403649
Train Epoch : 5 [57600/60000(96%)]      Train Loss : 0.571107

[Epoch : 5],    Test Loss : 0.0150,     Test Accuracy : 85.76%

Train Epoch : 6 [0/60000(0%)]   Train Loss : 0.395582
Train Epoch : 6 [6400/60000(11%)]       Train Loss : 0.369489
Train Epoch : 6 [12800/60000(21%)]      Train Loss : 0.392978
Train Epoch : 6 [19200/60000(32%)]      Train Loss : 0.596245
Train Epoch : 6 [25600/60000(43%)]      Train Loss : 0.219962
Train Epoch : 6 [32000/60000(53%)]      Train Loss : 0.323644
Train Epoch : 6 [38400/60000(64%)]      Train Loss : 0.497299
Train Epoch : 6 [44800/60000(75%)]      Train Loss : 0.474432
Train Epoch : 6 [51200/60000(85%)]      Train Loss : 0.457872
Train Epoch : 6 [57600/60000(96%)]      Train Loss : 0.370617

[Epoch : 6],    Test Loss : 0.0132,     Test Accuracy : 87.88%

Train Epoch : 7 [0/60000(0%)]   Train Loss : 0.234508
Train Epoch : 7 [6400/60000(11%)]       Train Loss : 0.521743
Train Epoch : 7 [12800/60000(21%)]      Train Loss : 0.339328
Train Epoch : 7 [19200/60000(32%)]      Train Loss : 0.649923
Train Epoch : 7 [25600/60000(43%)]      Train Loss : 0.222594
Train Epoch : 7 [32000/60000(53%)]      Train Loss : 0.491204
Train Epoch : 7 [38400/60000(64%)]      Train Loss : 0.658848
Train Epoch : 7 [44800/60000(75%)]      Train Loss : 0.318188
Train Epoch : 7 [51200/60000(85%)]      Train Loss : 0.284692
Train Epoch : 7 [57600/60000(96%)]      Train Loss : 0.416584

[Epoch : 7],    Test Loss : 0.0121,     Test Accuracy : 88.79%

Train Epoch : 8 [0/60000(0%)]   Train Loss : 0.247623
Train Epoch : 8 [6400/60000(11%)]       Train Loss : 0.866372
Train Epoch : 8 [12800/60000(21%)]      Train Loss : 0.265504
Train Epoch : 8 [19200/60000(32%)]      Train Loss : 0.441274
Train Epoch : 8 [25600/60000(43%)]      Train Loss : 0.457626
Train Epoch : 8 [32000/60000(53%)]      Train Loss : 0.388151
Train Epoch : 8 [38400/60000(64%)]      Train Loss : 0.410405
Train Epoch : 8 [44800/60000(75%)]      Train Loss : 0.194856
Train Epoch : 8 [51200/60000(85%)]      Train Loss : 0.516320
Train Epoch : 8 [57600/60000(96%)]      Train Loss : 0.494840

[Epoch : 8],    Test Loss : 0.0113,     Test Accuracy : 89.64%

Train Epoch : 9 [0/60000(0%)]   Train Loss : 0.243048
Train Epoch : 9 [6400/60000(11%)]       Train Loss : 0.804239
Train Epoch : 9 [12800/60000(21%)]      Train Loss : 0.237638
Train Epoch : 9 [19200/60000(32%)]      Train Loss : 0.307098
Train Epoch : 9 [25600/60000(43%)]      Train Loss : 0.263186
Train Epoch : 9 [32000/60000(53%)]      Train Loss : 0.712979
Train Epoch : 9 [38400/60000(64%)]      Train Loss : 0.482342
Train Epoch : 9 [44800/60000(75%)]      Train Loss : 0.501509
Train Epoch : 9 [51200/60000(85%)]      Train Loss : 0.356542
Train Epoch : 9 [57600/60000(96%)]      Train Loss : 0.188112

[Epoch : 9],    Test Loss : 0.0109,     Test Accuracy : 89.97%

Train Epoch : 10 [0/60000(0%)]  Train Loss : 0.402946
Train Epoch : 10 [6400/60000(11%)]      Train Loss : 0.375172
Train Epoch : 10 [12800/60000(21%)]     Train Loss : 0.280411
Train Epoch : 10 [19200/60000(32%)]     Train Loss : 0.269194
Train Epoch : 10 [25600/60000(43%)]     Train Loss : 0.211829
Train Epoch : 10 [32000/60000(53%)]     Train Loss : 0.548340
Train Epoch : 10 [38400/60000(64%)]     Train Loss : 0.369495
Train Epoch : 10 [44800/60000(75%)]     Train Loss : 0.228130
Train Epoch : 10 [51200/60000(85%)]     Train Loss : 0.346599
Train Epoch : 10 [57600/60000(96%)]     Train Loss : 0.242560

[Epoch : 10],   Test Loss : 0.0105,     Test Accuracy : 90.29%
"""