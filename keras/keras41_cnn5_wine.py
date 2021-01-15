import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

datasets=load_wine()
x=datasets.data
y=datasets.target

print(x.shape)
print(y.shape)

y=y.reshape(-1 ,1)

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x=x.reshape(x.shape[0], x.shape[1], 1, 1)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=45)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=45)

print(x_train.shape)
print(y_train.shape)

input1=Input(shape=(x_train.shape[1], 1, 1))
cnn1=Conv2D(150, (2,2), padding='same')(input1)
drop1=Dropout(0.2)(cnn1)
cnn1=Conv2D(200, (2,2), padding='same')(drop1)
drop1=Dropout(0.2)(cnn1)
flat1=Flatten()(drop1)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(150, activation='relu')(dense1)
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
output1=Dense(3, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=30, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss, acc : ', loss)
print('y_pred : \n', np.argmax(y_pred[:5], axis=-1))
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results
# loss, acc :  [0.13635282218456268, 0.9722222089767456]
# y_pred :
#  [2 2 0 1 1]
# y_test :
#  [2 2 0 1 1]