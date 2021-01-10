import numpy as np

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
(x_train, y_train), (x_test, y_test)=boston_housing.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1,1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1],1,1)

model=Sequential()
model.add(Conv2D(150, kernel_size=(2,2), padding='same', strides=1, input_shape=(13,1,1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(200, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
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

early=EarlyStopping(monitor='loss', patience=50, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


print(loss)
print(RMSE(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(y_pred[:5])
print(y_test[:5])

# results
# 15.168837547302246
# 3.8947193027813274
# 0.817778265695519
# [[ 8.215335]
#  [18.168524]
#  [20.885612]
#  [35.168137]
#  [24.139807]]
# [ 7.2 18.8 19.  27.  22.2]