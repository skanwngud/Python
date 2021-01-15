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

x=dataset[:,:4] # [행, 열] => [모든 행 : 첫 번째부터 4번째 열까지]
y=dataset[:,4]  # [행, 열] => [모든 행, 5번째 열]

# print(dataset)
# print(x)
# print(y)
# print(x.shape) # (6,4)
# print(y.shape) # (6, )

x=x.reshape(6,4,1)

model=Sequential()
model.add(LSTM(150, activation='relu', input_shape=(4,1)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))

ealry=EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, callbacks=ealry)

loss=model.evaluate(x,y)
pred=model.predict(x)

print(loss)
print(pred)

# results
# 6.464752004831098e-06
# [[ 5.0039277]
#  [ 5.9994383]
#  [ 6.9977913]
#  [ 7.9972034]
#  [ 8.999022 ]
#  [10.003064 ]]