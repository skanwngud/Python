import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train.reshape(x_train.shape[0], 32*32*3)
x_test=x_test.reshape(x_test.shape[0], 32*32*3)
x_val=x_val.reshape(x_val.shape[0], 32*32*3)

y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_val=y_val.reshape(-1, 1)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

print(x_train.shape)
print(y_train.shape)

model=Sequential()
model.add(Dense(200, activation='relu', input_shape=(32*32*3,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : \n', np.argmax(y_pred[:5], axis=-1))
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results
# loss :  [2.397040367126465, 0.4311000108718872]
# y_pred :
#  [3 8 0 8 4]
# y_test :
#  [3 8 8 0 6]