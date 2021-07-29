import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506, )
print('='*20)
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names) # column name

# MinMaxScalar = min~max/max-(min/max) => 0<=x<=1
# e.g. 0~711 > 0~711/711-0
# e.g. 100~711 > 100~711/711-(100/711)

# 데이터 전처리 (MinMax) - 성능 향상을 위해 반드시 해야함
x=x/711. 
# 데이터의 형변환 때문에 (float 형)
# x=(x-최소)/(최대-최소)
#  =(x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)

# 모델 구성
input1=Input(shape=13)
dense1=Dense(60, activation='relu')(input1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
output1=Dense(1, activation='relu')(dense1)
model=Model(inputs=input1, outputs=output1)

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=130, batch_size=1, validation_split=0.2)

# 평가, 예측
loss=model.evaluate(x_test, y_test, batch_size=1)
pred=model.predict(x_test)

def RMSE(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))

rmse=RMSE(y_test, pred)
r2=r2_score(y_test, pred)

print(loss)
print(rmse)
print(r2)

# 전처리 전
# [13.635822296142578, 2.712141990661621]
# 3.6926712892525257
# 0.8368587933642436


# 전처리 후
# [12.10439682006836, 2.588151454925537]
# 3.479137453964179
# 0.855180999735316 - 노드 35개

