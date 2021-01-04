# keras21_cancer1.py 를 다중분류로 코딩

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

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

print(y_train.shape)


input1=Input(shape=30)
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(2, activation='softmax')(dense1)
model=Model(inputs=input1, outputs=output1)

early_stopping=EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early_stopping)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])
print(np.argmax(y_pred, axis=-1))

# results
# [1.5441147089004517, 0.9473684430122375]
# [[0.0000000e+00 1.0000000e+00]
#  [1.0000000e+00 1.2750456e-13]
#  [0.0000000e+00 1.0000000e+00]
#  [2.4160601e-36 1.0000000e+00]]
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]
# [1 0 1 1]