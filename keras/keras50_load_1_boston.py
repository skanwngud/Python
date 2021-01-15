import numpy as np

x=np.load('../data/npy/boston_x.npy')
y=np.load('../data/npy/boston_y.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=33)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1))

es=EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=(x_val, y_val), callbacks=es)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print(loss)
print(RMSE(y_test, y_pred))
print(r2_score(y_test, y_pred))

# results
# 16.02338981628418
# 4.002922811283981
# 0.776317304022594