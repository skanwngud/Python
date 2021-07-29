# keras67_1 파일에 잡음 넣어서 복구
# numpy 파일 로드

import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import random

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, \
    Flatten, Activation, BatchNormalization

x_train=np.load('c:/data/image/data/train_set.npy')
x_test=np.load('c:/data/image/data/val_set.npy')

# print(x_train.shape) # (1389, 128, 128, 3)
# print(x_test.shape) # (347, 128, 128, 3)

x_train_noised=x_train+np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised=x_test+np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised=np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised=np.clip(x_test_noised, a_min=0, a_max=1)

model=Sequential()
model.add(Conv2D(64, 2, input_shape=(128, 128, 3), activation='relu', padding='same'))
model.add(Conv2D(3, 2, padding='same', activation='sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics='acc'
)

model.fit(
    x_train_noised, x_train,
    epochs=3
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
    ax.imshow(x_test[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i==0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i==0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i==0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout() # 찾아볼 것 : subplot 의 여백에 관한 모양을 지정해줌
plt.show()