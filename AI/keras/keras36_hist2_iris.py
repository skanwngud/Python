# categorical classification

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_iris()
x=dataset.data
y=dataset.target


from tensorflow.keras.utils import to_categorical
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

# 2. Modeling
input1=Input(shape=4)
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(inputs=input1, outputs=output1)

# 3. Compile, fitting
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist=model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# 4. Evaluate
loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])
print(np.argmax(y_pred, axis=-1))

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss, acc')
plt.xlabel('loss, acc')
plt.ylabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()
# results
# [0.11219839006662369, 0.9666666388511658]
# [[1.6016660e-10 6.5629654e-05 9.9993432e-01]
#  [1.0000000e+00 6.0144376e-11 3.4863581e-26]
#  [9.9662763e-01 3.3723419e-03 5.3760271e-11]
#  [6.4497421e-05 2.8200355e-01 7.1793193e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]