import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.transforms.transforms import ToTensor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device is {}'.format(device))
# current device is cuda

batch_size = 32
epochs = 10

train_data = datasets.FashionMNIST(
    root='data',
    download=True,
    train=True,
    transform=ToTensor(),                               # image data 를 tensor 로 바꿔주며, 0~255 픽셀값을 0~1로 정규화 (AE 는 input data 의 크기가 크면 성능이 제대로 안 나옴)
)

test_data = datasets.FashionMNIST(
    root='data',
    download=True,
    train=False,
    transform=ToTensor(),
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)

for (x_train, y_train) in train_loader:
    print('x_train : ', x_train.size(), 'type : ', x_train.type())
    print('y_train : ', y_train.size(), 'type : ', y_train.type())
    break
# x_train :  torch.Size([32, 1, 28, 28]) type :  torch.FloatTensor
# y_train :  torch.Size([32]) type :  torch.LongTensor

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(x_train[i, :, : ,:].numpy().reshape(28, 28), cmap='gray_r')
    plt.title('class : ' + str(y_train[i].item()))
plt.show()

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(                                   # 인코더를 정의하는 부분, 기존에는 모델을 하나하나 정의했지만 여기서는 한 번에 정의
            nn.Linear(28 * 28 , 512),                                   # input data 의 크기가 28 * 28
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

        self.decoder = nn.Sequential(                                   # 디코더를 정의하는 부분, 인코더와 반대 모야으로 정의
            nn.Linear(32, 256),                                         # 인코더의 최종 아웃풋 크기가 32 였으므로 디코더의 인풋은 32 가 된다
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),                                    # 인코더의 인풋 크기가 28 * 28 이었으므로 디코더의 아웃풋 역시 28 * 28
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)                                 # 인코더의 아웃풋을 디코더의 인풋으로 받는다
        return encoded, decoded                                         # forward propagation 을 거친 인코더와 디코더의 값을 반환한다

model = AE().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2
)
criterion = nn.MSELoss()                                                # 아웃풋 값은 label 과 같은 이미지 데이터 그 자체이기 때문에, MSE loss 를 사용

print(model)
# AE(
#   (encoder): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=256, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=256, out_features=32, bias=True)
#   )
#   (decoder): Sequential(
#     (0): Linear(in_features=32, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=784, bias=True)
#   )
# )

def train(model, train_loader, otpimizer, log_interval):
    model.train()

    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.view(-1, 28 * 28).to(device)
        target = image.view(-1, 28 * 28).to(device)

        optimizer.zero_grad()
        encoded, decoded = model(image)
        loss=criterion(decoded, target)
        loss.backward()
        otpimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Epochs : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f})%] Train Loss : {loss.item():.4f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    real_image = list()                                                 # 학습 과정 중에 이용 되는 실제 이미지 데이터를 저장하기 위한 리스트
    gen_image = list()                                                  # 학습 과정 속에 AE 를 통해 생성 되는 이미지 데이터를 저장하기 위한 리스트

    with torch.no_grad():
        for image, _ in test_loader:                                    # AE 에서는 label 은 사용하지 않으므로 _ 처리
            image = image.view(-1, 28 * 28).to(device)                  # 실제 이미지 데이터를 장치에 할당 (train 함수에서 정의한 것과 같은 모양 (-1, 28 * 28))
            target = image.view(-1, 28 * 28).to(device)                 # AE 를 통해 생성 되는 이미지 데이터를 장치에 할당 (train 함수에서 정의한 것과 같은 모양)
            encoded, decoded = model(image)                             # model 을 통해 반환 되는 값

            test_loss += criterion(decoded, image).item()
            real_image.append(image.to('cpu'))                          # 실제 이미지를 real_image 리스트에 저장
            gen_image.append(decoded.to('cpu'))                         # AE 를 통해 생성 된 이미지를 gen_image 리스트에 저장

        test_loss /= len(test_loader.dataset)
        return test_loss, real_image, gen_image

for epoch in range(1, epochs+1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, real_image, gen_image = test(model, test_loader)
    print(f'\nEpooch : {epoch} \tTest Loss : {test_loss:.4f}\n')
    f, a = plt.subplots(2, 10, figsize = (10, 4))
    for i in range(10):
        img = np.reshape(real_image[0][i], (28, 28))
        a[0][i].imshow(img, cmap = 'gray_r')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(10):
        img = np.reshape(gen_image[0][i], (28, 28))
        a[1][i].imshow(img, cmap = 'gray_r')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    plt.show()

"""
Epochs : 1 [0/60000 (0)%] Train Loss : 0.2526
Epochs : 1 [6400/60000 (11)%] Train Loss : 0.0394
Epochs : 1 [12800/60000 (21)%] Train Loss : 0.0344
Epochs : 1 [19200/60000 (32)%] Train Loss : 0.0288
Epochs : 1 [25600/60000 (43)%] Train Loss : 0.0256
Epochs : 1 [32000/60000 (53)%] Train Loss : 0.0278
Epochs : 1 [38400/60000 (64)%] Train Loss : 0.0310
Epochs : 1 [44800/60000 (75)%] Train Loss : 0.0297
Epochs : 1 [51200/60000 (85)%] Train Loss : 0.0264
Epochs : 1 [57600/60000 (96)%] Train Loss : 0.0255

Epooch : 1      Test Loss : 0.0008

Epochs : 2 [0/60000 (0)%] Train Loss : 0.0309
Epochs : 2 [6400/60000 (11)%] Train Loss : 0.0256
Epochs : 2 [12800/60000 (21)%] Train Loss : 0.0217
Epochs : 2 [19200/60000 (32)%] Train Loss : 0.0228
Epochs : 2 [25600/60000 (43)%] Train Loss : 0.0263
Epochs : 2 [32000/60000 (53)%] Train Loss : 0.0217
Epochs : 2 [38400/60000 (64)%] Train Loss : 0.0292
Epochs : 2 [44800/60000 (75)%] Train Loss : 0.0258
Epochs : 2 [51200/60000 (85)%] Train Loss : 0.0279
Epochs : 2 [57600/60000 (96)%] Train Loss : 0.0252

Epooch : 2      Test Loss : 0.0008

Epochs : 3 [0/60000 (0)%] Train Loss : 0.0247
Epochs : 3 [6400/60000 (11)%] Train Loss : 0.0216
Epochs : 3 [12800/60000 (21)%] Train Loss : 0.0260
Epochs : 3 [19200/60000 (32)%] Train Loss : 0.0239
Epochs : 3 [25600/60000 (43)%] Train Loss : 0.0227
Epochs : 3 [32000/60000 (53)%] Train Loss : 0.0213
Epochs : 3 [38400/60000 (64)%] Train Loss : 0.0226
Epochs : 3 [44800/60000 (75)%] Train Loss : 0.0253
Epochs : 3 [51200/60000 (85)%] Train Loss : 0.0218
Epochs : 3 [57600/60000 (96)%] Train Loss : 0.0231

Epooch : 3      Test Loss : 0.0007

Epochs : 4 [0/60000 (0)%] Train Loss : 0.0240
Epochs : 4 [6400/60000 (11)%] Train Loss : 0.0257
Epochs : 4 [12800/60000 (21)%] Train Loss : 0.0256
Epochs : 4 [19200/60000 (32)%] Train Loss : 0.0201
Epochs : 4 [25600/60000 (43)%] Train Loss : 0.0249
Epochs : 4 [32000/60000 (53)%] Train Loss : 0.0230
Epochs : 4 [38400/60000 (64)%] Train Loss : 0.0231
Epochs : 4 [44800/60000 (75)%] Train Loss : 0.0304
Epochs : 4 [51200/60000 (85)%] Train Loss : 0.0208
Epochs : 4 [57600/60000 (96)%] Train Loss : 0.0235

Epooch : 4      Test Loss : 0.0008

Epochs : 5 [0/60000 (0)%] Train Loss : 0.0218
Epochs : 5 [6400/60000 (11)%] Train Loss : 0.0304
Epochs : 5 [12800/60000 (21)%] Train Loss : 0.0228
Epochs : 5 [19200/60000 (32)%] Train Loss : 0.0268
Epochs : 5 [25600/60000 (43)%] Train Loss : 0.0202
Epochs : 5 [32000/60000 (53)%] Train Loss : 0.0260
Epochs : 5 [38400/60000 (64)%] Train Loss : 0.0227
Epochs : 5 [44800/60000 (75)%] Train Loss : 0.0224
Epochs : 5 [51200/60000 (85)%] Train Loss : 0.0253
Epochs : 5 [57600/60000 (96)%] Train Loss : 0.0285

Epooch : 5      Test Loss : 0.0007

...

Epochs : 10 [0/60000 (0)%] Train Loss : 0.0239
Epochs : 10 [6400/60000 (11)%] Train Loss : 0.0259
Epochs : 10 [12800/60000 (21)%] Train Loss : 0.0270
Epochs : 10 [19200/60000 (32)%] Train Loss : 0.0229
Epochs : 10 [25600/60000 (43)%] Train Loss : 0.0256
Epochs : 10 [32000/60000 (53)%] Train Loss : 0.0242
Epochs : 10 [38400/60000 (64)%] Train Loss : 0.0227
Epochs : 10 [44800/60000 (75)%] Train Loss : 0.0218
Epochs : 10 [51200/60000 (85)%] Train Loss : 0.0218
Epochs : 10 [57600/60000 (96)%] Train Loss : 0.0244

Epooch : 10     Test Loss : 0.0007
"""