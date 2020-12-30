import numpy as np

x1=np.array([range(100), range(301, 401), range(1, 101)])
y1=np.array([range(711, 811), range(1, 101), range(201, 301)])

x2=np.array([range(101, 201), range(411, 511), range(100,200)])
y2=np.array([range(501,601), range(711, 811), range(100)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test=train_test_split(x1, y1, shuffle=False, train_size=0.8)
x2_train, x2_test, y2_train, y2_test=train_test_split(x2, y2, shuffle=False, train_size=0.8)

# Sequential 은 앙상블에서 쓰기 힘들다 (순차적인 모델이기 때문에 상하관계가 되어버림)
# Sequential1, Sequential2 를 작성하고 돌리게 되면 1의 가중치가 2에 영향을 끼침

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model 1
input1=Input(shape=(3,))
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(5, activation='relu')(dense1)
# output1=Dense(3)(dense1)

# model 2
input2=Input(shape=(3,))
dense2=Dense(10, activation='relu')(input2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
dense2=Dense(5, activation='relu')(dense2)
# output2=Dense(3)(dense2)
# output 은 위의 모델들이 아니라 모델끼리 엮인 다음 나와야한다

## 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1=concatenate([dense1, dense2])
# 각 모델들의 끝 쪽에 있는 레이어들
# merge1 = dense1, dense2 를 인풋으로 받는 레이어 모델
middle1=Dense(30)(merge1)
middle1=Dense(10)(middle1)
middle1=Dense(10)(middle1)

## 모델 분기 1
output1=Dense(30)(middle1)
output1=Dense(7)(output1)
output1=Dense(3)(output1)

## 모델 분기 2
output2=Dense(15)(middle1)
output2=Dense(7)(output2)
output2=Dense(7)(output2)
output2=Dense(3)(output2)

# 모델 선언
model=Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)

loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)

print('model.metrics_name : ', model.metrics_names)
# model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mae', 'dense_15_mae']

print(loss)
# [901.44287109375, 344.4125671386719, 557.0303955078125, 344.4125671386719, 557.0303955078125] - loss=mse, metrics=mse
# 두 모델 loss 합, 첫 번째 모델 loss, 두 번째 모델 loss, 첫 번째 모델 metrics, 두 번째 모델 metrics
# [2611.45751953125, 1208.328125, 1403.1292724609375, 29.186691284179688, 29.323776245117188] - loss=mse, metrics=mae

y1_predict, y2_predict=model.predict([x1_test, x2_test])

print('y1_pred : \n', y1_predict)
print('y2_pred : \n', y2_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE_1(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))

def RMSE_2(y2_test, y2_predict):
    return np.sqrt(mean_squared_error(y2_test, y2_predict))

r2_1=r2_score(y1_test, y1_predict)
r2_2=r2_score(y2_test, y2_predict)

print('rmse1 : ',RMSE_1)
print('rmse2 : ',RMSE_2)
print('r2_1 : ',r2_1)
print('r2_2 : ',r2_2)