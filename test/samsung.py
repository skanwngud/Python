import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('../data/csv/samsung.csv', encoding='euc-kr',header=0, index_col=0, thousands=',')

# print(df)
# print(df.shape) # (2400, 14)

# 데이터 날짜 오름차순으로 변경
df=df.sort_index(ascending=True)
# 액면분할 이후 시점부터 슬라이싱
df=df.iloc[1738:, :6]

# 넘파이로 변경 
df=df.to_numpy()
# print(type(df[1,1])) # numpy.float64
# print(df.shape) # (662, 6)
df_1=df[:5, :]
print(df_1)
print(df_1.shape)

def split_x (seq, size):
    aaa=[]
    for i in range(len(seq-size+1)):
        subset=seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

size=5

# print(df)
df=split_x(df, size)
# print(df)
# df_1=df[:,]
print(df.shape)
# print(df[:2])

# input=Input(shape=(6,1))
# lstm1=LSTM(150, activation='relu')(input)
# drop1=Dropout(0.2)(lstm1)
# dense1=Dense(100, activation='relu')(drop1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(200, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(120, activation='relu')(dense1)
# dense1=Dense(10, activation='relu')(dense1)
# output=Dense(1)(dense1)
# model=Model(input, output)

# model.compile(loss='mse', optimizer='adam', metrics='mae')
# model.fit()