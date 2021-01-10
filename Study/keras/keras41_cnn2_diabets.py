import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=56)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)

model=Sequential()
model.add(Conv2D(150, (2,2), padding='same', input_shape=(x_val.shape[1], 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

early=EarlyStopping(monitor='loss', patience=30, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=early)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ',loss)
print('rmse : ',RMSE(y_test, y_pred))
print('r2 : ',r2_score(y_test, y_pred))
print('y_pred : \n',y_pred[:5])
print('y_test : \n',y_test[:5])

# results
# loss :  4922.77734375
# rmse :  70.1625066874448
# r2 :  0.11966626503557631
# y_pred :
#  [[117.04573]
#  [ 91.08257]
#  [122.13173]
#  [249.98676]
#  [100.85487]]
# y_test :
#  [ 48. 131. 101. 277. 108.]