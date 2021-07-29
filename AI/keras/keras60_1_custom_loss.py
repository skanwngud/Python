## import library
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## make function
def custom_mse(y_true, y_pred): # 훈련 과정 중에 자동적으로 인자를 받아 넣어줌
    return tf.math.reduce_mean(tf.square(y_true-y_pred)) # mse

## data
x=np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8, )
y=np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8, )

## model
model=Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

## compile, fitting
model.compile(loss=custom_mse, optimizer='adam')
model.fit(x,y, batch_size=1, epochs=30)

## evaluate, predict
loss=model.evaluate(x,y,)

print(loss)