import numpy as np

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test)=cifar100.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train.reshape(x_train.shape[0], 32*4, 24)/255.
x_test=x_test.reshape(x_test.shape[0], 32*4, 24)/255.
x_val=x_val.reshape(x_val.shape[0], 32*4, 24)/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

input1=Input(shape=(32*4, 24))
lstm1=LSTM(150, activation='relu')(input1)
drop1=Dropout(0.2)(lstm1)
dense1=Dense(128, activation='relu')(drop1)
dense1=Dense(64, activation='relu')(dense1)
dense1=Dense(32, activation='relu')(dense1)
dense1=Dense(64, activation='relu')(dense1)
output1=Dense(100, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : ', np.argmax(y_pred[:5], axis=-1))
print('y_test : ', np.argmax(y_test[:5], axis=-1))

# results
# loss :  [4.568361759185791, 0.018200000748038292]
# y_pred :  [52 52 52 52 43]
# y_test :  [49 33 72 51 71]