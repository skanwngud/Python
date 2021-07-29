import numpy as np

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train.reshape(x_train.shape[0], 32*8, 12)
x_test=x_test.reshape(x_test.shape[0], 32*8, 12)
x_val=x_val.reshape(x_val.shape[0], 32*8, 12)

y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_val=y_val.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

model=Sequential()
model.add(LSTM(128, activation='relu', input_shape=(32*8, 12)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

early=EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : ', np.argmax(y_pred[:5], axis=-1))
print('y_test : ', np.argmax(y_test[:5], axis=-1))

# results
# loss :  [2.3026201725006104, 0.10000000149011612]
# y_pred :  [4 4 4 4 4]
# y_test :  [3 8 8 0 6]