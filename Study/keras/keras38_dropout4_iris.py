# categorical classification

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

# 1. Data
dataset=load_iris()
x=dataset.data
y=dataset.target

from tensorflow.keras.utils import to_categorical

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

# 2. Modeling
input1=Input(shape=4)
dense1=Dense(150, activation='relu')(input1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(inputs=input1, outputs=output1)

# 3. Compile, fitting
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# 4. Evaluate
loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])
print(np.argmax(y_pred, axis=-1))

# results
# [0.11219839006662369, 0.9666666388511658]
# [[1.6016660e-10 6.5629654e-05 9.9993432e-01]
#  [1.0000000e+00 6.0144376e-11 3.4863581e-26]
#  [9.9662763e-01 3.3723419e-03 5.3760271e-11]
#  [6.4497421e-05 2.8200355e-01 7.1793193e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]

# dropout - acc 는 동일, mse 는 줄었음
# [0.09937789291143417, 0.9666666388511658]
# [[2.8895851e-14 1.7055601e-04 9.9982941e-01]
#  [1.0000000e+00 3.9877343e-10 6.8772127e-18]
#  [9.9975795e-01 2.4204144e-04 1.5475544e-08]
#  [1.7653801e-07 6.9951415e-02 9.3004841e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]