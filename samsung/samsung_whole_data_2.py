import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv('../data/csv/samsung_df_whole_data.csv',header=0, index_col=0, thousands=',')
df_1=pd.read_csv('../data/csv/samsung_df_1_whole_data.csv', header=0, index_col=0, thousands=',')
df_2=pd.read_csv('../data/csv/samsung2_whole_data.csv', encoding='cp949', header=0, index_col=0, thousands=',')

df_2=df_2.sort_index(ascending=True)
df_2=df_2.dropna(axis=0)
df_2=df_2.iloc[:2, :6]

df_2['target']=df_2.iloc[:, 3]
del df_2['종가']

df=pd.concat([df, df_2])

df=df.drop([df.columns[6], df.columns[7]], axis=1)

df=df.fillna(method='ffill')

'''
df.to_csv('../data/csv/samsung_df_day2.csv', sep=',')
'''

df=df.to_numpy()
df_1=df_1.to_numpy()

df_x=df[:, :-1]
df_y=df[:, -1]
df_1_x=df_1[:, :-1]
df_1_y=df_1[:, -1]

scaler_1, scaler_2=MinMaxScaler(), MinMaxScaler()
scaler_1.fit(df_x)
df_x=scaler_1.transform(df_x)
scaler_2.fit(df_1_x)
df_1_x=scaler_2.transform(df_1_x)

print(df_x)
print(df_1_x)

df_y=df_y.reshape(-1, 1)
df_1_y=df_1_y.reshape(-1, 1)

df_y=np.vstack([df_1_y, df_y])
df_x=np.vstack([df_1_x, df_x])

df=np.hstack([df_x, df_y])

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

x=df[:-1, :, :-1]
y=df[1:, -1:, -1:]
x_pred=df[-1:, :, :-1]

x=x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred=x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
y=y.reshape(y.shape[0], 1)

# scaler=MinMaxScaler()
# scaler.fit(x)
# x=scaler.transform(x)
# x_pred=scaler.transform(x_pred)

x=x.reshape(x.shape[0], 5, 5)
x_pred=x_pred.reshape(x_pred.shape[0], 5, 5)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=33)

'''
# np.savez('../data/npy/samsung_data_2.npz',
#         x_train=x_train, x_test=x_test, x_val=x_val, x_pred=x_pred,
#         y_train=y_train, y_test=y_test, y_val=y_val)
'''

np.savez('../data/npy/samsung_data_2(whole_data).npz',
        x_train=x_train, x_test=x_test, x_val=x_val, x_pred=x_pred,
        y_train=y_train, y_test=y_test, y_val=y_val)

model=Sequential()
model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

cp=ModelCheckpoint(filepath='../data/modelcheckpoint/samsung_day2_{val_loss:.4f}.hdf5',
                    monitor='val_loss', mode='auto', save_best_only=True)
es=EarlyStopping(monitor='val_loss', mode='auto', patience=50)
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), callbacks=[es, cp], verbose=2)

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

pred=model.predict(x_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
print('samsung_predict : ', pred)
