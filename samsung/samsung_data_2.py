import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

df_1=pd.read_csv('../data/csv/samsung_df.csv',header=0, index_col=0, thousands=',')
df_2=pd.read_csv('../data/csv/samsung2.csv', encoding='cp949', header=0, index_col=0, thousands=',')

# print(df_2.shape) # (60, 16)

df_2=df_2.sort_index(ascending=True)
df_2=df_2.dropna(axis=0)
df_2=df_2.iloc[:2, :6]

df_2['target']=df_2.iloc[:, 3]
del df_2['종가']

df=pd.concat([df_1, df_2])

df=df.drop([df.columns[6], df.columns[7]], axis=1)
# print(df_2.shape) # (2, 6)

df=df.fillna(method='ffill')
# print(df.columns)
# print(df.info())

df.to_csv('../data/csv/samsung_df_day2.csv', sep=',')

df=df.to_numpy()

# print(df.shape) # (664, 6)

# print(df)


def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

data=df
label=df
size=5
col=6

df=split_x(df, size, col)

# print(df.shape) # (660, 5, 6)

x=df[:-1, :, :-1]
y=df[1:, -1:, -1:]
x_pred=df[-1:, :, :-1]

# print(x.shape) # (659, 5, 5)
# print(y.shape) # (659, 1, 1)
# print(x_pred.shape) # (1, 5, 5)

x=x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred=x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
y=y.reshape(y.shape[0], 1)

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x_pred=scaler.transform(x_pred)

x=x.reshape(x.shape[0], 5, 5)
x_pred=x_pred.reshape(x_pred.shape[0], 5, 5)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=22)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=22)

# print(x_train.shape) # (421, 5, 5)
# print(x_test.shape) # (132, 5, 5)
# print(x_val.shape) # (106, 5, 5)
# print(y_train.shape) # (412, 1)
# print(y_test.shape) # (132, 1)
# print(y_val.shape) # (106, 1)
# print(x_pred.shape) # (1, 5, 5)

np.savez('../data/npy/samsung_data_2.npz',
        x_train=x_train, x_test=x_test, x_val=x_val, x_pred=x_pred,
        y_train=y_train, y_test=y_test, y_val=y_val)

model=Sequential()
model.add(LSTM(256, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

cp=ModelCheckpoint(filepath='../data/modelcheckpoint/samsung_day2_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', mode='auto', save_best_only=True)
es=EarlyStopping(monitor='val_loss', mode='auto', patience=10)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[es, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

pred=model.predict(x_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
print('samsung_predict : ', pred)