# sklearn / LSTM - binary classfication

# sklearn / LSTM - regression

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

dataset=load_breast_cancer()

x=dataset.data
y=dataset.target

y=y.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

print(y_train.shape)

print(x_train.shape)

model=Sequential()
model.add(LSTM(256, activation='relu', input_shape=(30, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[:5])

print(loss)
print(y_pred)
print(np.argmax(y_pred, axis=-1))