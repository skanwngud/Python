import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Flatten, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train=pd.read_csv('./dacon/train/train.csv', header=0, index_col=0)
sub=pd.read_csv('./dacon/sample_submission.csv', header=0, index_col=0)

df_test = []

for i in range(81):
    file_path = './dacon/test/' + str(i) + '.csv'
    temp=pd.read_csv(file_path)
    df_test.append(temp)

x_test = pd.concat(df_test)

print(x_train.info())
print(x_test.info())

df_train=x_train.iloc[:, 2:]
df_test=x_test.iloc[:, 3:]

print(df_train.info())

df_test.to_csv('./dacon/dacon_test.csv', sep=',')

df_train=df_train.to_numpy()
df_test=df_test.to_numpy()

print(df_train.shape) # (52560, 6)

df_train=df_train.reshape(-1, 48, 6)

print(df_train.shape) # 데이터를 하루치씩 나눔

def split_x(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col-1
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y=split_x(df_train, 7, 3)

print(x.shape) # (1087, 7, 48, 6)
print(y.shape) # (1087, 2, 48, 6)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=False)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, shuffle=False)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2]*x_val.shape[3])

mms=MinMaxScaler()
mms.fit(x_train)
x_train=mms.transform(x_train)
x_testt=mms.transform(x_test)
x_val=mms.transform(x_val)

x_train=x_train.reshape(x_train.shape[0], 7, 48 ,6)
x_test=x_test.reshape(x_test.shape[0], 7, 48, 6)
x_val=x_val.reshape(x_val.shape[0], 7, 48, 6)

print(x_train.shape) # (695, 7, 48, 6)
print(x_test.shape) # (218, 7, 48, 6)
print(x_val.shape) # (174, 7, 48, 6)

print(y_train.shape) # (659, 2, 48, 6)
print(y_test.shape) # (218, 2, 48, 6)
print(y_val.shape) # (174, 2, 48, 6)

model=Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(7, 48, 6)))
model.add(Dense(128, activation='relu'))
model.add(Dense(6))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(y_pred)