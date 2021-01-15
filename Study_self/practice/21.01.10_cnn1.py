import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=44)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)/255.

y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_val=y_val.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input1=Input(shape=(x_train.shape[1], x_train.shape[2], 1))
cnn1=Conv2D(150, (2,2), padding='same')(input1)
pool1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(pool1)
cnn1=Conv2D(200, (2,2))(drop1)
pool1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(pool1)
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
dense1=Dense(15, activation='relu')(dense1)
output1=Dense(10, activation='softmax')(dense1)
model=Model(input1, output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss, acc : ', loss)
print('y_pred : ', y_pred[:5])
print('y_test : ', y_test[:5])
print('y_test : ', np.argmax(y_test[:5], axis=-1))

# results
# loss, acc :  [0.07134567201137543, 0.9858999848365784]
# y_pred :  [[2.5379117e-18 2.8377154e-15 7.2987560e-10 4.9422422e-10 5.1295353e-12
#   7.2657761e-19 1.0329172e-22 1.0000000e+00 2.2205835e-18 4.4663573e-10]
#  [3.4349977e-15 8.9871114e-18 9.9999988e-01 6.1960932e-08 1.5526224e-22
#   1.5800093e-17 1.5719005e-23 1.1298718e-08 3.3661989e-13 4.1102071e-18]
#  [4.7189786e-13 9.9999952e-01 6.8250010e-08 8.6625294e-09 1.6660465e-08
#   5.8351936e-16 1.7854314e-11 3.0768282e-07 2.7619819e-08 1.7444064e-09]
#  [1.0000000e+00 9.1431807e-16 6.2109962e-10 5.2337430e-11 8.9037274e-15
#   2.0271753e-09 3.6040920e-08 6.0910360e-11 1.6892233e-08 1.2281517e-09]
#  [3.3294705e-17 2.8245525e-10 5.9014369e-12 1.8801157e-17 9.9999440e-01
#   1.2117023e-15 1.7549397e-09 5.6401500e-10 1.3955384e-12 5.5783439e-06]]
# y_test :  [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
# y_test :  [7 2 1 0 4]