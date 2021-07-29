import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_val=y_val.reshape(-1, 1)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

# enc=OneHotEncoder()
# enc.fit(y_train)
# y_train=enc.transform(y_train).toarray()
# y_test=enc.transform(y_test).toarray()
# y_val=enc.transform(y_val).toarray()

input1=Input(shape=(x_train.shape[1], x_train.shape[2], 1))
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
dense1=Dense(350, activation='relu')(dense1)
dense1=Dense(300, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(50, activation='relu')(dense1)
output1=Dense(10, activation='softmax')(dense1)
model=Model(input1, output1)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : \n', np.argmax(y_pred[:5], axis=-1))
# print('y_test : \n', np.argmax(y_test[:5], axis=-1))
print('y_test : \n', y_test[:5])

# results - categorical_crossentropy
# loss :  [0.2846837341785431, 0.9210000038146973]
# y_pred :
#  [7 2 1 0 4]
# y_test :
#  [7 2 1 0 4]

# results - sparse_categorical_crossentropy, batch_size=64
# loss :  [0.17495787143707275, 0.9656999707221985]
# y_pred :
#  [7 2 1 0 4]
# y_test :
#  [[7]
#  [2]
#  [1]
#  [0]
#  [4]]