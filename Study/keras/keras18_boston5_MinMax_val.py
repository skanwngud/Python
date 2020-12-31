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

print(np.max(x[0])) # 396.9

print(np.max(x), np.min(x)) # 711.0 0.0 -> 1.0 0.0
print(np.max(x[0])) # 0.9999999999999999

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

# 모델 구성
input1=Input(shape=13)
dense1=Dense(35, activation='relu')(input1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
dense1=Dense(35, activation='relu')(dense1)
output1=Dense(1, activation='relu')(dense1)
model=Model(inputs=input1, outputs=output1)

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val))

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

# 전처리 후 - x=x/711.
# [12.10439682006836, 2.588151454925537]
# 3.479137453964179
# 0.855180999735316

# 전처리 후 - MinMaxScaler (x)
# [8.861637115478516, 2.0854930877685547]
# 2.9768507285717583
# 0.8939778794202871

# 전처리 후 - MinMaxScaler (x_train, validation_split)
# [16.556724548339844, 2.515528917312622]
# 4.068996103714728
# 0.8019125694719111

# 전처리 후 - MinMaxScaler (x_train, validation_data)
# [5.61604118347168, 1.900649070739746]
# 2.3698186392016685
# 0.93280876980413

# [6.636787414550781, 1.8860915899276733]
# 2.5761964757264995
# 0.9205963741714427 - epochs = 200

# [7.60515832901001, 2.147317886352539]
# 2.7577447488751563
# 0.9090106624303158 - epochs = 1000, batch_size=8