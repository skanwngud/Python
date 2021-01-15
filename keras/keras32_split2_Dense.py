# DNN 구성


import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

a=np.array(range(1,11))
size=5

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset=split_x(a, size)

x=dataset[:,:4]
y=dataset[:,4]

# print(dataset)
# print(x)
# print(y)
# print(x.shape) # (6,4)
# print(y.shape) # (6, )

# x=x.reshape(6,4,1)

model=Sequential()
model.add(Dense(150, activation='relu', input_shape=(4,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

ealry=EarlyStopping(monitor='loss', patience=50, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, callbacks=ealry)

loss=model.evaluate(x,y)
pred=model.predict(x)

print(loss)
print(pred)

# results - LSTM
# 6.464752004831098e-06
# [[ 5.0039277]
#  [ 5.9994383]
#  [ 6.9977913]
#  [ 7.9972034]
#  [ 8.999022 ]
#  [10.003064 ]]

# results - Dense
# 1.4493377875623992e-07
# [[ 5.000145 ]
#  [ 6.0004582]
#  [ 6.9993496]
#  [ 7.99969  ]
#  [ 9.000018 ]
#  [10.000345 ]]