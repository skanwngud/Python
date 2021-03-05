# import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,\
    Activation, BatchNormalization, Input, UpSampling2D
from keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test)=mnist.load_data()

# check data shape
# print(x_train.shape, y_train.shape) # (60000, 28, 28) , (60000, )
# print(x_test.shape, y_test.shape) # (10000, 28, 28), (10000, )

# reshape data
x_train=x_train.reshape(-1, 28, 28, 1)/255. # preprocessing (0~255 -> 0~1)
x_test=x_test.reshape(-1, 28, 28, 1)/255.

# modeling
input=Input(shape=(28, 28, 1))
x=Conv2D(64, 2, padding='same', activation='relu')(input)
x=MaxPooling2D(2, padding='same')(x)
x=Conv2D(64, 2, padding='same', activation='relu')(x)
x=MaxPooling2D(2, padding='same')(x)
x=Conv2D(32, 2, padding='same', activation='relu')(x)
x=Conv2D(32, 2, padding='same', activation='relu')(x)
x=UpSampling2D()(x)
x=Conv2D(64, 2, padding='same', activation='relu')(x)
x=UpSampling2D()(x)
x=Conv2D(64, 2, padding='same', activation='relu')(x)
output=Conv2D(1, 2, padding='same', activation='sigmoid')(x)
model=Model(input, output)

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

model.fit(
    x_train, x_train,
    validation_split=0.2,
    epochs=10,
    batch_size=256
)

loss=model.evaluate(
    x_test, x_test
)

results=model.predict(
    x_test
)

# random image generate
random_imgs=np.random.randint(x_test.shape[0], size=5) # x_test.shape[0] : x_test 의 갯수, size=5 : 5개의 이미지를 뽑아냄

plt.figure(figsize=(7, 2)) # 출력 이미지 크기 정하기

for i, img_idx in enumerate(random_imgs):
    ax=plt.subplot(2, 7, i+1) # 원본 이미지를 출력 할 크기
    plt.imshow(x_test[img_idx].reshape(28, 28)) # 원본 이미지를 보여줌
    ax.axis('off')
    ax=plt.subplot(2, 7, 7+i+1) # 오토인코딩 된 이미지를 출력 할 크기
    plt.imshow(results[img_idx].reshape(28, 28)) # 오토인코딩 된 이미지를 보여줌
    ax.axis('off')

plt.show()