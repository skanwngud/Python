import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
# 하나하나는 모듈. 전체는 폴더 구조 자체로 되어있는 것을 패키지

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) #(28,28)
print(np.max)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
# x_train의 최댓값 : 255
# .astype('float32') -> 정수형을 실수형으로 바꾸는 것
x_test= x_test.reshape(10000,28,28,1)/255. 
# 이렇게 해도 실수형으로 바로 된다. 
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) -> 코딩할 때 이렇게 쓰기!

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#epoch:02d 정수형으로 2자리까지 표현, .4f는 소수 4번째자리까지 나온다.
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# filepath :  최저점마다 파일을 생성하는데 파일 안에 그 지점의 w값이 들어간다. predict, evaluate 할 때 파일에서 땡겨쓰면 좋다. 가장 마지막이 제일 값이 좋은 것.

########################### 모델부터 컴파일까지 gpu 분산처리 ###############
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(9,2))
    model.add(Conv2D(8,2))
    model.add(Dropout(0.2))
    model.add(Flatten()) 
    model.add(Dense(40,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#########################################################################
hist = model.fit(x_train,y_train, callbacks=[es], epochs=2, validation_split=0.2, batch_size=16)

# 기본은 모델이 완벽해야 모델체크포인트에 저장된게 좋은것. 모델이 안좋으면 쓰레기 안에서 그나마 나은걸 뽑은 것.

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747