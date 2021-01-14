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

def split_x(seq, label, size):
    data=[]
    labe=[]
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
# print(x)
# print(y)

scaler=MinMaxScaler()
scaler.fit(x)
df=scaler.transform(x)

# print(x.shape) # (661, 5)
# print(y.shape) # (661, )
# print(x_pred)
# print(x_pred.shape) # (1, 5)
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=44)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=44)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# print(x_train)
# print(x_test)
# print(x_val)
# print(x_pred)
# print(y_train)
# print(y_test)
# print(y_val)

np.save('../data/npy/samsung_x_train.npy', arr=x_train)
np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_x_val.npy', arr=x_val)
np.save('../data/npy/samsung_y_train.npy', arr=y_train)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)
np.save('../data/npy/samsung_y_val.npy', arr=y_val)
np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)


input=Input(shape=(x_train.shape[1], 1))
lstm1=LSTM(200, activation='relu')(input)
# drop1=Dropout(0.2)(lstm1)
dense1=Dense(200, activation='relu')(lstm1)
dense1=Dense(220, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(220, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
output=Dense(1)(dense1)
model=Model(input, output)

# input=Input(shape=(x_train.shape[1], 1))
# cnn1=Conv1D(200, 5, padding='same', activation='relu')(input)
# max1=MaxPooling1D(2)(cnn1)
# drop1=Dropout(0.2)(max1)
# cnn1=Conv1D(200, 5, padding='same', activation='relu')(drop1)
# max1=MaxPooling1D(2)(cnn1)
# drop1=Dropout(0.2)(max1)
# flat1=Flatten()(drop1)
# dense1=Dense(100, activation='relu')(flat1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(200, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(100, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# output=Dense(1)(dense1)
# model=Model(input,output)

es=EarlyStopping(monitor='val_loss', patience=100, mode='auto')
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

plt.figure(figsize=(10, 6))

plt.title('val_loss')
plt.xlabel('epchs')
plt.ylabel('val_loss')
plt.hist(hist.history['val_loss'])

plt.show()