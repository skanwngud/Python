from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

x=np.array(range(1,101))
y=np.array(range(1,101))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=False)
# test_size=0.2 == train_size=0.8 (둘이 반대되는 개념)
# x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False)
#   - sum of size 가 1을 넘어가면 안 된다는 메세지 출력과 함께 돌아가지 않음

# x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False)
#   - 위의 데이터 기준 train=(1~70), test=(71~90) 으로 나눠짐

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8)

print(x_train)
print(x_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

model=Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

loss, mae=model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict=model.predict(x_test)
print(y_predict)

# shuffle=False
# loss :  0.07874681800603867
# mae :  0.2774181365966797

# shuffle=True
# loss :  0.01689854823052883
# mae :  0.11417099088430405

# validation = 0.2
# loss :  0.0007629828178323805
# mae :  0.024267006665468216

# validation_data
# loss :  0.010812943801283836
# mae :  0.08748173713684082
# rmse :  0.1039853096987895
# r2 : 0.9999835846660394

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print('rmse : ', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)

print('r2 :', r2)
