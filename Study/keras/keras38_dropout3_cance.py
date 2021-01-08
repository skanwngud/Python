# binary classification

import numpy as np

from sklearn.datasets import load_breast_cancer

# 1. Data
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

# 2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model=Sequential()
model.add(Dense(135, activation='relu', input_shape=(30,)))
# relu - 범위가 0 부터 무한대로 수렴 (0 이하의 값은 0 으로 받음)
# linear - 범위가 마이너스 무한대부터 무한대까지
# hidden layer 에 sigmoid 를 써도 상관은 없지만 연산량이 줄어도 성능이 떨어질 수도 있다
# (무조건 떨어지는 것은 아니며 상황에 따라 다름)
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile, fitting
                # = mean_squared_error (mse)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# evaluate, predict
loss=model.evaluate(x_test,y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])

print(np.where(y_pred<0.5, 0, 1))

# y_list=list()
# for i in y_pred:
#     if i >= 0.5:
#         y_list.append(1)
#     else:
#         y_list.append(0)
# print(y_list)

# results
# [1.086754322052002, 0.9561403393745422]
# [[1.0000000e+00]
#  [1.7427337e-36]
#  [1.0000000e+00]
#  [1.0000000e+00]]
# [1 0 1 1]
# [1, 0, 1, 1]

# dropout
# [2.4364383220672607, 0.9736841917037964]
# [[1.000000e+00]
#  [6.016992e-20]
#  [1.000000e+00]
#  [1.000000e+00]]
# [1 0 1 1]
# [[1]
#  [0]
#  [1]
#  [1]]