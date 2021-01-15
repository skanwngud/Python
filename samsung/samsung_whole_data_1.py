import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

df=pd.read_csv('../data/csv/samsung.csv', encoding='euc-kr',header=0, index_col=0, thousands=',')
df_1=pd.read_csv('../data/csv/samsung.csv', encoding='euc-kr', header=0, index_col=0, thousands=',')

df=df.sort_index(ascending=True)
df=df.dropna(axis=0)
df_1=df_1.sort_index(ascending=True)
df_1=df.dropna(axis=0)

df=df.iloc[1735:, :6] # (663, 6)
df_1=df_1.iloc[:1735,:6]

# x = 5/4~1/12
# y = 5/8~1/13
# x_pred = 1/13

df['target']=df.iloc[:, 3]
del df['종가']

df_1['target']=df_1.iloc[:, 3]
del df_1['종가']

df.to_csv('../data/csv/samsung_df_whole_data.csv', sep=',')
df_1.to_csv('../data/csv/samsung_df_1_whole_data.csv', sep=',')

df=df.to_numpy()
df_1=df_1.to_numpy()

# print(df) 
# print(df.shape) # (662, 6)
# print(df_1.shape) # (1735, 6)

df_x=df[:, :-1]
df_y=df[:, -1]
df_1_x=df_1[:, :-1]
df_1_y=df_1[:, -1]

scaler_1=MinMaxScaler()
scaler_1.fit(df_x)
df_x=scaler_1.transform(df_x)
df_1_x=scaler_1.transform(df_1_x)

df_y=df_y.reshape(-1, 1)
df_1_y=df_1_y.reshape(-1, 1)

df_y=np.vstack([df_1_y, df_y])
df_x=np.vstack([df_1_x, df_x])

print(df_x.shape)
print(df_y.shape)

df=np.hstack([df_x, df_y])

print(df.shape)

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

data=df_1
label=df_1

df=split_x(df, size, col)
df_1=split_x(df_1, size, col)
# print(df.shape) # (658, 5, 6)
# print(df)
x=df[:-1, :, :-1] # 컬럼값 (657, 5, 5)
y=df[1:, -1:, -1:] # 종가값 (657, 1, 1)
x_pred=df[-1:, :, :-1] # 예측값 (1, 5, 5)

print(x.shape)
print(y.shape)
print(x_pred.shape)

x=x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred=x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
y=y.reshape(y.shape[0], 1)
'''
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x_pred=scaler.transform(x_pred)
'''
x=x.reshape(x.shape[0], 5, 5)
x_pred=x_pred.reshape(x_pred.shape[0],5, 5)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=44)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=44)

'''
np.save('../data/npy/samsung_x_train.npy', arr=x_train)
np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_x_val.npy', arr=x_val)
np.save('../data/npy/samsung_y_train.npy', arr=y_train)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)
np.save('../data/npy/samsung_y_val.npy', arr=y_val)
np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)
'''

np.savez('../data/npy/samsung_data_1_whole_data.npz',
        x_train=x_train, x_test=x_test, x_val=x_val, x_pred=x_pred,
        y_train=y_train, y_test=y_test, y_val=y_val)
input=Input(shape=(x_train.shape[1], x_train.shape[2]))
lstm1=LSTM(256, activation='relu')(input)
dense1=Dense(256, activation='relu')(lstm1)
dense1=Dense(256, activation='relu')(dense1)
dense1=Dense(512, activation='relu')(dense1)
dense1=Dense(512, activation='relu')(dense1)
dense1=Dense(1024, activation='relu')(dense1)
dense1=Dense(512, activation='relu')(dense1)
dense1=Dense(512, activation='relu')(dense1)
dense1=Dense(256, activation='relu')(dense1)
dense1=Dense(256, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(64, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='val_loss', patience=30, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/samsung_{val_loss:.4f}.hdf5', monitor='val_loss', mode='auto', save_best_only=True)
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=2000, batch_size=64, validation_data=(x_val, y_val), verbose=2, callbacks=[es, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

pred=model.predict(x_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse=RMSE(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print('loss : ', loss)
print('samsung_predict :', pred)
print('rmse : ', rmse)
print('r2 : ', r2)

plt.plot(hist.history['loss'], marker='.', c='blue')
plt.plot(hist.history['val_loss'], marker='.', c='red')
plt.grid()

plt.title('loss & val_loss')
plt.xlabel('epoch')
plt.ylabel('loss & val_loss')
plt.legend(['loss', 'val_loss'])

plt.show()
