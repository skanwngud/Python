import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101), range(501, 601), range(651, 751)])
y=np.array([range(711, 811), range(101, 201)])

x_pred2=np.array([100, 401, 101, 601, 751]) # (5,)

print(x.shape) # (5, 100)
print(y.shape) # (2, 100)
print(x_pred2.shape)

x=np.transpose(x)
y=np.transpose(y)
x_pred2=x_pred2.reshape(1, 5)

print(x.shape) # (100, 5)
print(y.shape) # (100, 2)
print(x_pred2.shape) 

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.2, shuffle=True, random_state=66)

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 2)

model=Sequential()
model.add(Dense(20, input_dim=5, activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=130, batch_size=1, validation_split=0.2, verbose=1)

'''
verbose=0 훈련과정을 보여주지 않음
verbose=1 훈련과정 전체를 보여줌
verbose=2 진행막대를 보여주지 않음
verbose=3 epochs만 보여줌
(verbose=3 이후부터는 결과값 같음)

디폴트는 verbose=1
'''

# 학습의 경과 시간이 긴 모델의 경우 훈련과정을 터미널창에 찍는 동안 생기는 딜레이가 누적이 되어 학습기간이 더 길어질 수 있으므로,
# 0으로 두거나 필요없는 과정을 생략함으로써 시간을 조금 더 단축시킬 수 있다

loss, mae=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2=r2_score(y_test, y_predict)

print('rmse : ', RMSE(y_test, y_predict))
print('loss : ', loss)
print('mae : ', mae)
print('r2 : ', r2)

y_pred2=model.predict(x_pred2)
print(y_pred2)