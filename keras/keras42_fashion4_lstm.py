import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train.reshape(x_train.shape[0], 28*7, 4)
x_test=x_test.reshape(x_test.shape[0], 28*7, 4)
x_val=x_val.reshape(x_val.shape[0], 28*7, 4)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

input1=Input(shape=(28*7, 4))
lstm1=LSTM(100, activation='relu')(input1)
drop1=Dropout(0.2)(lstm1)
dense1=Dense(100, activation='relu')(drop1)
drop1=Dropout(0.2)(dense1)
dense1=Dense(150, activation='relu')(drop1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(300, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output1=Dense(10, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=3, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : \n', np.argmax(y_pred[:5], axis=-1))
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results
# loss :  [2.3026180267333984, 0.10000000149011612]
# y_pred :
#  [2 2 2 2 2]
# y_test :
#  [9 2 1 1 6]