random_seed=66

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
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


# modelcheckpoint
import datetime
import time
import os
# 현재 시간 출력

filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath=''.join([filepath, 'k45_', '{timer}', filename])

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        try:
            file_path=self.filepath.format(epoch=epoch+1, timer=datetime.datetime.now().strftime('%m%d_%H%M'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath:"{}".'
                'Reason:{}'.format(self.filepath, e))
        self._write_filepath=distributed_file_utils.write_filepath(file_path, self.model.distribute_strategy)
        return self._write_filepath



'''
# print(date_now) # 2021-01-27 10:06:07.219552 / 실제 시간이 아닌 컴퓨터에 설정 된 시간으로 표기해줌

# print(date_time) # 0127_1010
filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
date_now=datetime.datetime.now() # 문제점 : 이 시간대로 고정이 됨
# date_time=date_now.strftime('%m%d_%H%M') # %m = month, %d = day, %H = Hour, %M = Minute
date_time=date_now.strftime('%x_%X') # 01/27/21_18:16:11
modelpath=os.path.join(filepath, 'k45_', date_time, filename) # .join 문자열 합치는 함수 

print(date_time)
print(type(date_time))
# print(modelpath)

# print(modelpath)
# print(timen_n())
# modelpath='..\data\modelcheckpoint\k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
# skanwngud\Study\modelCheckpoint
'''

early=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp=MyModelCheckpoint(filepath=modelpath,
                  monitor='val_loss', save_best_only=True, mode='auto',)
# cp=ModelCheckpoint(filepath='../data/modelcheckpoint/k45_modelcheckpoint_{epoch:02d}-{val_loss:.4f}.hdf5',
#                     monitor='val_loss', save_best_only=True, mode='auto')
# filepath - 가중치 세이브, 최저점을 찍을 때마다 weight 가 들어간 파일을 만듬
# 세이브 된 최적 가중치를 이용해서 모델 평가, 예측을 좀 더 쉽고 빠르게 할 수 있다


# compile, fitting
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist=model.fit(x_train, y_train, epochs=30, batch_size=256, validation_split=0.2, callbacks=[early, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))

'''
plt.figure(figsize=(10, 6)) # (10, 6) 의 면적을 잡음

plt.subplot(2,1,1) # (2, 1) 짜리 그림을 만듬 (2행 1열의 그림 중 '첫 번째')
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 격자의 형태

plt.title('Cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') # loc = location, loc 를 명시하지 않으면 빈 공간에 들어감

plt.subplot(2,1,2) # (2, 1) 짜리 그림을 만듬 (2행 1열의 그림 중 '두 번째')
plt.plot(hist.history['acc'], marker='.', c='red') # model.compile 에서 accuracy 로 썼으면 여기서도 풀네임으로 써야한다
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc']) # plt.plot 에서 label 을 지정해주지 않아도 plt.legend 에 직접 넣어줄 수 있다

plt.show()

# results
# [0.05556643009185791, 0.9904000163078308]
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]
'''