from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

x=np.array(range(1,101))
y=np.array(range(1,101))

# x_train=x[:60]
# x_val=x[60:80]
# x_test=x[80:]

# y_train=y[:60]
# y_val=y[60:80]
# y_test=y[80:]

# 데이터 전처리 과정 등에서 필요한 툴은 sklearn 에 많이 있으므로, 자주 활용
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, shuffle=True)
# train_size=0.6 == x 의 데이터 중 60%를 트레인셋으로 만듬 (무작위)
# shuffle=False 로 하면 순서대로 데이터가 슬라이싱 됨 (True가 default)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

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
model.fit(x_train, y_train, epochs=100)

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
