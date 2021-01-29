## import library
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## make function
def custom_mse(y_true, y_pred): # 훈련 과정 중에 자동적으로 인자를 받아 넣어줌
    return tf.math.reduce_mean(tf.square(y_true-y_pred)) # mse

def quantile_loss(y_true, y_pred):
    qs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q=tf.constant(np.array([qs]), dtype=tf.float32) # constant = 상수
    e=y_true, y_pred
    v=tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def quantile_loss_dacon(q, y_true, y_pred):
    err=(y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

## data
x=np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8, )
y=np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8, )

## model
for q in quantiles:
    model=Sequential()
    model.add(Dense(10, input_shape=(1,)))
    model.add(Dense(10))
    model.add(Dense(1))

## compile, fitting
# model.compile(loss=quantile_loss, optimizer='adam')
    model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles, y_true, y_pred), optimizer='adam')
    model.fit(x,y, batch_size=1, epochs=30)

    ## evaluate, predict
    loss=model.evaluate(x,y,)

print(loss)

# mse
# 0.0008114440133795142

# quantile_loss
# 1.1306196451187134

# quantile_loss[0]
# 0.002973938127979636

'''
LGBMRgressor
objective = quantile_loss - LGBM 에서 이미 퀀타일로스를 제공한다
alpha = quantile_list
'''