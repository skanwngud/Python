import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

dataset=load_iris()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_val=y_val.reshape(-1, 1)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

enc=OneHotEncoder()
enc.fit(y_train.reshape(-1, 1))
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

print(y_train.shape) # (96, 3)
print(x_train.shape) # (96, 4)

model=Sequential()
model.add(Dense(150, activation='relu', input_shape=(4,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(3, activation='softmax'))

early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early_stopping)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])
print(np.argmax(y_pred, axis=-1))

# results
# [0.08190993964672089, 0.9666666388511658]
# [[1.9900340e-12 2.0987708e-04 9.9979013e-01]
#  [9.9999952e-01 4.4907455e-07 4.7529017e-19]
#  [9.9944860e-01 5.5133947e-04 2.1601963e-10]
#  [1.3220053e-06 3.4845737e-01 6.5154135e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]