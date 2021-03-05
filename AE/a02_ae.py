import numpy as np
import tensorflow
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input

(x_train, _), (x_test, _)=mnist.load_data()

x_train=x_train.reshape(60000, 784).astype('float32')/255
x_test=x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])


def autoencoder(hidden_layer_size):
    model=Sequential()
    model.add(Dense(
        units=hidden_layer_size, 
        input_shape=(784,),
        activation='relu'))
    model.add(Dense(
        units=784, 
        activation='sigmoid'))

    return model

model=autoencoder(hidden_layer_size=154)
# hidden layer size 가 높을수록 결과값이 입력값이랑 비슷해진다


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
    x_train, x_train,
    epochs=10
)

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