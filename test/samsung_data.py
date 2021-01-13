import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

df=pd.read_csv('../data/csv/samsung.csv', encoding='euc-kr',header=0, index_col=0, thousands=',')

df=df.sort_index(ascending=True)
df=df.dropna(axis=0)

df=df.iloc[1735:, :6] # (663, 6)

# x = 5/4~1/12
# y = 5/8~1/13
# x_pred = 1/13

df['target']=df.iloc[:, 3]
del df['종가']

# print(df)
# print(df.iloc[:-1,:]) # x
# print(df.iloc[:-1,:].shape) # (661, 6)
# print(df.iloc[1:,-1:]) # y
# print(df.iloc[1:,-1:].shape) # (661, 1)
# print(df.iloc[-1:,:-1]) # pred

df=df.to_numpy()

def split_x(seq, label, size):
    data=[]
    label=[]
    for i in range(len(data)-size):
        data.append(seq[i:i+size])
        label.append(seq[i+size])
    return np.array(data), np.array(label)

data=df
label=df[:,-1:]
size=5

df=split_x(data, label, size)

x=data[:-1, :-1] # 컬럼값
y=label[1:, :] # 종가값
x_pred=data[-1:, :-1] # 예측값
print(x)
print(y)

scaler=MinMaxScaler()
scaler.fit(x)
df=scaler.transform(x)

print(x.shape) # (661, 5)
print(y.shape) # (661, )
print(x_pred)
print(x_pred.shape) # (1, 5)
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, shuffle=False)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, shuffle=False)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


np.save('../data/npy/samsung_x_train.npy', arr=x_train)
np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_x_val.npy', arr=x_val)
np.save('../data/npy/samsung_y_train.npy', arr=y_train)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)
np.save('../data/npy/samsung_y_val.npy', arr=y_val)
np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)


input=Input(shape=(x_train.shape[1], 1))
lstm1=LSTM(100, activation='relu')(input)
drop1=Dropout(0.2)(lstm1)
dense1=Dense(80, activation='relu')(drop1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', mode='auto', save_best_only=True)
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=5000, validation_data=(x_val, y_val), verbose=2, callbacks=[es, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_pred)

print(loss)
print(y_pred)