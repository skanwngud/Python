random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try :
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e) # GPU 분산



(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'))
# kernel_initializer : kernel == weight / weight 정규화
model.add(BatchNormalization())
# BatchNormalization : Batch 값을 정규화
model.add(Activation('relu'))

model.add(Conv2D(128, 3, padding='same', kernel_regularizer=l1(l1=0.01)))
# kernel_regularizer
model.add(Dropout(0.25))

model.add(Conv2D(128, 3, padding='same'))
model.add(MaxPooling2D(2))

model.add(Flatten())

model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

early=EarlyStopping(monitor='val_loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist=model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))

# results
# [0.05556643009185791, 0.9904000163078308]
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]

'''
kernel_initializer : He - relu, selu, elu ....
                     Xavier - sigmoid, tahn
(kernel : 가중치(weight))
                     kernel_initializer 를 하게 되면 얼마나 gradient 를 잘 전달 할 수 있느냐와
                     layer 를 얼마나 깊게 쌓을 수 있느냐가 정해짐
                     kernel_initializer 에 존재하는 he 와 xavier 는 각각
                     relu, selu, elu 등과 sigmoid, tahn 등에 사용할 때 적합하다

bias_initializer : bias 는 활성화 함수에 직접적으로 관여하게 되므로 몹시 중요한데,
                   기존에는 0.01 이나 0.1 처럼 매우 작은 양수를 주었으나,
                   학습 방법이 개선 된 지금은 보통 0 으로 초기화를 시킴

kernel_regularizer : 레이어 복잡도에 제한을 두어 가중치가 가장 작은 값을 가지도록 강제함
                     (가중치 값의 분포가 균일해짐)

BatchNormalization : 레이어에 들어가는 batch 값들을 정규화 시킴

Dropout : 훈련 할 때 node 의 갯수를 무작위로 줄임 / 검증할 때엔 dropout 을 하지 않음

Batch, Dropout 과 같이 쓰면 안 좋다고는 하지만 무조건 확정적인 것은 아니며,
실제로도 gan 에서도 함께 쓰이기도 한다
위 다섯가지 정리
'''