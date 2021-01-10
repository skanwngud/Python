import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

y=y.reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(y)
y=enc.transform(y).toarray()

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x=x.reshape(x.shape[0], x.shape[1], 1, 1)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=56)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

print(x_train.shape)
print(y_train.shape)

input1=Input(shape=(x_train.shape[1], 1, 1))
cnn1=Conv2D(150, (2,2), padding='same')(input1)
dropout1=Dropout(0.2)(cnn1)
cnn2=Conv2D(200, (2,2), padding='same')(dropout1)
dropout2=Dropout(0.2)(cnn2)
flatten1=Flatten()(dropout2)
dense1=Dense(100, activation='relu')(flatten1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output1=Dense(2, activation='sigmoid')(dense1)
model=Model(input1, output1)

early=EarlyStopping(monitor='loss', patience=30, mode='auto')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss, acc : ', loss)
print('y_pred : \n', y_pred[:5])
print('y_test : \n', np.argmax(y_test[:5], axis=-1))

# results
# loss, acc :  [0.03218197450041771, 0.9824561476707458]
# y_pred :
#  [[1.0000000e+00 1.7577203e-16]
#  [1.4858405e-05 9.9997675e-01]
#  [1.9830234e-04 9.9971408e-01]
#  [5.3126394e-13 1.0000000e+00]
#  [1.0000000e+00 4.0899882e-31]]
# y_test :
#  [0 1 1 1 0]