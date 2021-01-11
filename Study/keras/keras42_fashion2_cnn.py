import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

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

print(x_train.shape)
print(y_train.shape)


input1=Input(shape=(x_train.shape[1], x_train.shape[2], 1))
cnn1=Conv2D(200, (2,2), padding='same')(input1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(250, (2,2), padding='same')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
flat1=Flatten()(max1)
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
output1=Dense(10, activation='softmax')(dense1)
model=Model(input1, output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : \n', np.argmax(pred[:5], axis=-1))
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

plt.imshow(y_test[0], 'gray')
plt.show

# results
# loss :  [0.31359612941741943, 0.9103000164031982]
# y_pred :
#  [9 2 1 1 6]
# y_test :
#  [9 2 1 1 6]