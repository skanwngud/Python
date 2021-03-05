# a04 번 복사
# 딥러닝 모델로 구성 (기준점을 기준으로 동일하게)
# CAE = Convolution AutoEncoder

import numpy as np
import tensorflow
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Flatten, UpSampling2D

(x_train, _), (x_test, _)=mnist.load_data()

x_train=x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test=x_test.reshape(10000, 28, 28, 1)/255.
x_train_2=x_train.reshape(60000, 784)/255.
x_test_2=x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])


def autoencoder():
    model=Sequential()
    model.add(Conv2D(154, 2, padding='same' ,input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten())
    # model.add(UpSampling2D(size=(1,1)))
    model.add(Dense(784 ,activation='sigmoid'))

    return model

def autoencoder():
    model=Sequential()
    model.add(Conv2D(154, 2, padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(1, 2, padding='same', activation='sigmoid'))

model=autoencoder()

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='acc'
)

# model.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics='acc'
# )

model.fit(
    x_train, x_train_2,
    epochs=10
)

loss=model.evaluate(
    x_test, x_test_2
)

print('loss : ', loss[0])
print('acc : ', loss[1])

output=model.predict(
    x_test
)

import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))=\
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯개를 무작위로 선별
random_images=random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨 위에 작성
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i==0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i==0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

# results

# conv2d
# loss :  0.0035317561123520136
# acc :  0.0421999990940094