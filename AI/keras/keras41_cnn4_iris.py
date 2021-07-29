import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

datasets=load_iris()
x=datasets.data
y=datasets.target

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x=x.reshape(x.shape[0], x.shape[1], 1, 1)

# y=to_categorical(y)
y=y.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=56)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

print(x_train.shape)
print(y_train.shape)

input1=Input(shape=(x_train.shape[1], 1, 1))
cnn1=Conv2D(150, (2,2), padding='same')(input1)
drop1=Dropout(0.2)(cnn1)
cnn2=Conv2D(200, (2,2), padding='same')(drop1)
drop2=Dropout(0.2)(cnn2)
flat1=Flatten()(drop2)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
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
print('y_pred : \n', y_pred[:5])
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results
# loss, acc :  [0.09535675495862961, 0.9666666388511658]
# y_pred :
#  [[1.9217872e-08 3.3727080e-02 9.6627289e-01]
#  [1.5977131e-02 9.8376691e-01 2.5603309e-04]
#  [1.8023389e-06 6.3858789e-01 3.6141026e-01]
#  [4.1439530e-10 4.9586329e-03 9.9504143e-01]
#  [7.8481055e-11 3.1385950e-03 9.9686146e-01]]
# y_test :
#  [2 1 1 2 2]