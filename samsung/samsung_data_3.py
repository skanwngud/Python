# 코스닥은 컬럼 6개 이상


## 임포트
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, concatenate, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


## csv 로드
df=pd.read_csv('../data/csv/samsung.csv', encoding='cp949', header=0, index_col=0, thousands=',')
df_1=pd.read_csv('./samsung/samsung2.csv', encoding='cp949', header=0, index_col=0, thousands=',')
df_2=pd.read_csv('../data/csv/삼성전자0115.csv', encoding='cp949', header=0, index_col=0, thousands=',')
df_3=pd.read_csv('../data/csv/KODEX 코스닥150 선물인버스.csv', encoding='cp949', header=0, index_col=0, thousands=',')

# plt.rc('font', family='Malgun Gothic')
# sns.heatmap(data=df.corr(), annot=True, cbar=True, square=True)
# plt.show()
# 시, 고, 저, 종, 프로그램, 개인, 기관

## 컬럼분리
df=df.iloc[:, [0,1,2,3,8,9,12]]
df_1=df_1.iloc[:1, [0,1,2,3,10,11,14]]
df_2=df_2.iloc[:1, [0,1,2,3,10,11,14]]
df_3=df_3.iloc[:, [0,1,2,3,10,11,14]]


## 날짜 오름차순으로 변경
df=df.iloc[:662, :]
df=df.sort_index(ascending=True)
df_3=df_3.iloc[:664, :]
df=df.sort_index(ascending=True)


## 날짜순으로 통합
df=pd.concat([df, df_1])
df=pd.concat([df, df_2])

# df=df.fillna(method='ffill')
# df_3=df_3.fillna(method='ffill')
## numpy 타입으로 변경
df=df.to_numpy()
df_3=df_3.to_numpy()

# print(df.shape) # (664, 7)
# print(df_3.shape) # (664, 7)
# print(type(df)) # numpy
# print(type(df_3)) # numpy


## 데이터 스플릿
def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

size=6
col=7

df=split_x(df, size, col)
df_3=split_x(df_3, size, col)

# print(df.shape) # (659, 5, 7)
# print(df_3.shape) # (659, 5, 7)


## 데이터 슬라이싱
x=df[:-1, :5, 1:]
y=df[1:, -1:, 0] # 하루 건너 치를 계산, y=df[1:, -2:, 0] 는 이틀치를 계산
x_pred=df[-1:, :5, 1:]

# print(x.shape) # (658, 5, 6)
# print(y.shape) # (658, 1)
# print(x_pred.shape) # (1, 5, 6)

x_1=df_3[:-1, :5, 1:]
y_1=df_3[1:, -1:, 0]
x_1_pred=df_3[-1:, :5, 1:]

# print(x_1.shape) # (658, 5, 6)
# print(y_1.shape) # (658, 1)
# print(x_pred_1.shape) # (1, 5, 6)


## 전처리
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=33)

x_1_train, x_1_test, y_1_train, y_1_test=train_test_split(x_1, y_1, train_size=0.8, random_state=33)
x_1_train, x_1_val, y_1_train, y_1_val=train_test_split(x_1_train, y_1_train, train_size=0.8, random_state=33)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])
x_pred=x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])

x_1_train=x_1_train.reshape(x_1_train.shape[0], x_1_train.shape[1]*x_1_train.shape[2])
x_1_test=x_1_test.reshape(x_1_test.shape[0], x_1_test.shape[1]*x_1_test.shape[2])
x_1_val=x_1_val.reshape(x_1_val.shape[0], x_1_val.shape[1]*x_1_val.shape[2])
x_1_pred=x_1_pred.reshape(x_1_pred.shape[0], x_1_pred.shape[1]*x_1_pred.shape[2])

scaler, scaler_1=MinMaxScaler(), MinMaxScaler()
scaler.fit(x_train)
scaler_1.fit(x_1_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)
x_pred=scaler.transform(x_pred)

x_1_train=scaler_1.transform(x_1_train)
x_1_test=scaler_1.transform(x_1_test)
x_1_val=scaler_1.transform(x_1_val)
x_1_pred=scaler_1.transform(x_1_pred)

x_train=x_train.reshape(x_train.shape[0], 5, 6)
x_test=x_test.reshape(x_test.shape[0], 5, 6)
x_val=x_val.reshape(x_val.shape[0], 5, 6)
x_pred=x_pred.reshape(x_pred.shape[0], 5, 6)

x_1_train=x_1_train.reshape(x_1_train.shape[0], 5, 6)
x_1_test=x_1_test.reshape(x_1_test.shape[0], 5, 6)
x_1_val=x_1_val.reshape(x_1_val.shape[0], 5, 6)
x_1_pred=x_1_pred.reshape(x_1_pred.shape[0], 5, 6)


## npy 파일 저장
np.savez('../data/npy/samsung_day_3.npz',
        x_train=x_train, x_test=x_test, x_val=x_val, x_pred=x_pred,
        y_train=y_train, y_test=y_test, y_val=y_val,
        x_1_train=x_1_train, x_1_test=x_1_test, x_1_val=x_1_val, x_1_pred=x_1_pred)


## 모델링
input1=Input(shape=(x_train.shape[1], x_train.shape[2]))
# lstm1=LSTM(32, activation='relu')(input1)
# dense1=Dense(64, activation='relu')(lstm1)
cnn1=Conv1D(32, 2, padding='same', activation='relu')(input1)
flat1=Flatten()(cnn1)
dense1=Dense(64, activation='relu')(cnn1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(256, activation='relu')(dense1)
dense1=Dense(128, activation='relu')(dense1)
dense1=Dense(64, activation='relu')(dense1)
dense1=Dense(32, activation='relu')(dense1)
dense1=Dense(32, activation='relu')(dense1)

input2=Input(shape=(x_1_train.shape[1], x_1_train.shape[2]))
lstm2=LSTM(32, activation='relu')(input2)
# lstm2=Dense(64, activation='relu')(input2)
dense2=Dense(128, activation='relu')(lstm2)
dense2=Dense(256, activation='relu')(dense2)
dense2=Dense(128, activation='relu')(dense2)
dense2=Dense(64, activation='relu')(dense2)
dense2=Dense(32, activation='relu')(dense2)
dense2=Dense(32, activation='relu')(dense2)

merge=concatenate([dense1, dense2])
mid1=Dense(64, activation='relu')(merge)
mid1=Dense(128, activation='relu')(mid1)
mid1=Dense(256, activation='relu')(mid1)
mid1=Dense(128, activation='relu')(mid1)
mid1=Dense(64, activation='relu')(mid1)
mid1=Dense(32, activation='relu')(mid1)
mid1=Dense(16, activation='relu')(mid1)
mid1=Dense(16, activation='relu')(mid1)

output=Dense(1)(mid1)

model=Model([input1, input2], output)


## 컴파일, 훈련
es=EarlyStopping(monitor='val_loss', mode='auto', patience=50)
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/Samsung_day_3_{val_loss:.4f}.hdf5',
                    monitor='val_loss', mode='auto', save_best_only=True)
model.compile(loss='mse', optimizer='adam')
model.fit([x_train, x_1_train], y_train, validation_data=([x_val, x_1_val], y_val),
            epochs=1500, batch_size=16, callbacks=[cp, es], verbose=2)


## 평가, 예측
loss=model.evaluate([x_test, x_1_test], y_test)
y_pred=model.predict([x_test, x_1_test])

pred=model.predict([x_pred, x_1_pred])


## 출력
print(loss)
print(pred)
print(r2_score(y_test, y_pred))
