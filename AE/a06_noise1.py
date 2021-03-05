
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import random

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Flatten, UpSampling2D

(x_train, _), (x_test, _)=mnist.load_data()

x_train=x_train.reshape(60000, 784).astype('float32')/255
x_test=x_test.reshape(10000, 784)/255.

# 노이즈 생성
x_train_noised=x_train+np.random.normal(0, 0.1, size=x_train.shape) # random.normal = 정규분포에 맞게 0부터 0.1 사이의 수치를 점으로 찍음
x_test_noised=x_test+np.random.normal(0, 0.1, size=x_test.shape) # x 의 값이 0~1 이기 때문에 0~1.1 의 범위를 갖게 되며 전체적으로 0.1 만큼 밝아짐
x_train_noised=np.clip(x_train_noised, a_min=0, a_max=1) # 최소 최대값을 0~1 로 고정 시켜줌
x_test_noised=np.clip(x_test_noised, a_min=0, a_max=1)

def autoencoder(hidden_layer_size):
    model=Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model=autoencoder(hidden_layer_size=154) # 특성의 95% 를 갖고 있는 수치 (그 미만이면 제대로 복원이 안 되고 그 이상이면 원본이미지와 차이가 안 남)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='acc'
)

model.fit(
    x_train_noised, x_train, # 노이즈가 있는 데이터와 없는 데이터를 번갈아가며 훈련시킴
    epochs=10
)

output=model.predict(
    x_test_noised
)

# 시각화
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15))=\
        plt.subplots(3, 5, figsize=(20, 7))


# 이미지 다섯개를 무작위로 선별
random_images=random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그림
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i==0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i==0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i==0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout() # 찾아볼 것 : subplot 의 여백에 관한 모양을 지정해줌
plt.show()

# 노이즈가 있는 데이터, 노이즈가 없는 데이터가 있어야 훈련 가능