import numpy as np
import tensorflow

from keras.datasets import mnist

(x_train, _), (x_test, _)=mnist.load_data() # y 값을 빼버림 / _ : 자릿수는 맞춰주되 데이터는 쓰지 않겠다는 의미

x_train=x_train.reshape(60000, 784).astype('float32')/255
x_test=x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img=Input(shape=(784,))
encoded=Dense(64, activation='relu')(input_img)
decoded=Dense(784, activation='sigmoid')(encoded)

autoencoder=Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='acc' # 지표는 acc 보다 loss 로 잡아야 좀 더 정확하다 (784, sigmoid) 이기 때문
)

# autoencoder.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics='acc'
# )

autoencoder.fit(
    x_train, x_train, # x와 y가 동일하기 때문에 둘 다 x_train 을 넣는다
    validation_split=0.2,
    epochs=30,
    batch_size=256
)

decoded_img=autoencoder.predict(
    x_test
)

import matplotlib.pyplot as plt

# 이미지 시각화
n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax=plt.subplot(2, n, i+1) # 위에 10개
    plt.imshow(x_test[i].reshape(28, 28)) # 원래 이미지
    plt.gray()
    ax.get_xaxis().set_visible(False) # 숫자 출력
    ax.get_yaxis().set_visible(False)

    ax=plt.subplot(2, n, i+1+n) # 아래에 10개
    plt.imshow(decoded_img[i].reshape(28, 28)) # decodend 된 이미지
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()