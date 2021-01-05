import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

ealry=EarlyStopping(monitor='loss', patience=20, mode='auto')

dataset=load_iris()
x=dataset.data
y=dataset.target
y=y.reshape(-1, 1)

y=to_categorical(y)
# one=OneHotEncoder()
# one.fit(y)
# y=one.transform(y).toarray()

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

print(x_train.shape)
print(y_train.shape)

input1=Input(shape=4)
dense1=Dense(256, activation='relu')(input1)
dense1=Dense(256, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(64, activation='relu')(dense1)
dense1=Dense(32, activation='relu')(dense1)
dense1=Dense(16, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
model=Model(inputs=input1, outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), callbacks=ealry)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test[10:15])

print('loss : ', loss)
print('y_pred : \n', y_pred)
print(np.argmax(y_pred, axis=-1))
print(y_test[10:15])

# # OneHotEncoder
# loss :  [0.23982693254947662, 0.8666666746139526]
# y_pred :  [[1.3251560e-04 2.9434627e-02 9.7043288e-01]
#  [1.6426937e-05 1.1151038e-02 9.8883259e-01]
#  [9.9986839e-01 1.3159614e-04 1.5849596e-13]
#  [2.9667228e-04 4.4506568e-02 9.5519680e-01]
#  [1.3731170e-04 2.8401826e-02 9.7146082e-01]]
# [2 2 0 2 2]
# [[0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]

# to_categorical
# loss :  [0.08726216852664948, 0.9666666388511658]
# y_pred :  [[2.15711020e-06 2.13983078e-02 9.78599548e-01]
#  [1.78374393e-07 4.56050551e-03 9.95439351e-01]
#  [9.99996662e-01 3.31555134e-06 1.18871502e-19]
#  [1.50447795e-05 1.17960654e-01 8.82024229e-01]
#  [3.66985159e-06 1.37420697e-02 9.86254215e-01]]
# [2 2 0 2 2]
# [[0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]