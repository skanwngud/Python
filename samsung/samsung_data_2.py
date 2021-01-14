import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

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

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
x_pred=scaler.transform(x_pred)

x=x.reshape(x.shape[0], 5, 5)
x_pred=x_pred.reshape(x_pred.shape[0],5,5)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=44)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=44)

# print(x_train)
# print(y_train)
# print(x_pred)
# print(y[0])

np.save('../data/npy/samsung_x_train.npy', arr=x_train)
np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_x_val.npy', arr=x_val)
np.save('../data/npy/samsung_y_train.npy', arr=y_train)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)
np.save('../data/npy/samsung_y_val.npy', arr=y_val)
np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)


# input=Input(shape=(x_train.shape[1], x_train.shape[2]))
# lstm1=LSTM(100, activation='relu')(input)
# # drop1=Dropout(0.2)(lstm1)
# dense1=Dense(120, activation='relu')(lstm1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(200, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# output=Dense(1)(dense1)
# model=Model(input, output)

input=Input(shape=(x_train.shape[1], x_train.shape[2]))
cnn1=Conv1D(200, 5, padding='same', activation='relu')(input)
max1=MaxPooling1D(2)(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv1D(200, 5, padding='same', activation='relu')(drop1)
drop1=Dropout(0.2)(cnn1)
cnn1=Conv1D(200, 5, padding='same', activation='relu')(drop1)
drop1=Dropout(0.2)(cnn1)
flat1=Flatten()(drop1)
dense1=Dense(60, activation='relu')(flat1)
dense1=Dense(80, activation='relu')(flat1)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(80, activation='relu')(flat1)
dense1=Dense(60, activation='relu')(flat1)
output=Dense(1)(dense1)
model=Model(input,output)

es=EarlyStopping(monitor='val_loss', patience=50, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/samsung_{val_loss:.4f}.hdf5', monitor='val_loss', mode='auto', save_best_only=True)
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), verbose=2, callbacks=[es, cp])

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

# plt.plot(hist.history['loss'], marker='.', c='blue')
# plt.plot(hist.history['val_loss'], marker='.', c='red')
# plt.grid()

# plt.title('loss & val_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss & val_loss')
# plt.legend(['loss', 'val_loss'])

# plt.show()