import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df_train=pd.read_csv('./dacon/train/train_1.csv', header=0, index_col=0)
df_test=pd.read_csv('./dacon/test/test_1.csv', header=0, index_col=0)

# print(df_train.info())
# print(df_test.info())

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

df_train=df_train.reshape(-1, 48, 8)
df_test=df_test.reshape(-1, 7, 48, 6)

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 2)

# print(x.shape) # (1087, 7, 48, 8)
# print(y.shape) # (1087, 2, 48, 8)
# print(df_test.shape) # (81, 7, 48, 6)

x=x[:, :, :, :6]
y=y[:, :, :, -2:]

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=False)

print(x_train.shape) # (869, 7, 48, 6)
print(y_train.shape) # (869, 2, 48, 2)

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(7,48,6)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2,
            epochs=100, batch_size=16)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(df_test)

print(loss)
print(y_pred.shape)