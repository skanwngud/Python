import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

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

input1=Input(shape=(28*28,))
dense1=Dense(100, activation='relu')(input1)
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
output1=Dense(10, activation='softmax')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, callbacks=early, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : \n', np.argmax(y_pred[:5], axis=-1))
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results - 71/100
# loss :  [0.5094953775405884, 0.8895999789237976]
# y_pred :
#  [9 2 1 1 6]
# y_test :
#  [9 2 1 1 6]

# results - 57/100
# loss :  [0.568520724773407, 0.8870000243186951]
# y_pred :
#  [9 2 1 1 6]
# y_test :
#  [9 2 1 1 6]