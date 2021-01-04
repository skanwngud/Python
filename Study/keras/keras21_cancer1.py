# binary classification

import numpy as np

from sklearn.datasets import load_breast_cancer

# 1. Data
datasets=load_breast_cancer()
# print(datasets.feature_names)
# print(datasets.DESCR) # Attribute = column

x=datasets.data
y=datasets.target

# print(x.shape) # (569, 30)
# print(y.shape) # (569, )
# print(x[:5])
# print(y) # 0 or 1

# preprocessing - MinMaxScaler, train_test_split
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
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(135, activation='relu', input_shape=(30,)))
# relu - 범위가 0 부터 무한대로 수렴 (0 이하의 값은 0 으로 받음)
# linear - 범위가 마이너스 무한대부터 무한대까지
# hidden layer 에 sigmoid 를 써도 상관은 없지만 연산량이 줄어도 성능이 떨어질 수도 있다
# (무조건 떨어지는 것은 아니며 상황에 따라 다름)
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# sigmoid - 범위를 0부터 1 사이로 제한
# 수업 중 activation, loss 등을 수정했을 때 수치가 최저 0.3에서 최대 0.9 까지 올랐는데,
# 이 경우 모델의 성능이 향상 된 것이 아닌 지표를 올바르게 수정했기 때문에 오른 것

# compile, fitting
                # = mean_squared_error (mse)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# 이진분류에서 metrics 는 '가급적' acc(accuracy)
# 이진분류에서는 현재로썬 '무조건' loss=binary_crossentropy
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# evaluate, predict
loss=model.evaluate(x_test,y_test)
y_pred=model.predict(x_test[-5:-1])

print(loss)
print(y_pred)
# print(y[-5:-1])

print(np.argmax(y_pred, axis=-1))

# results
# [0.5336576700210571, 0.9649122953414917]
# [[0.] - sigmoid 함수는 0, 1로만 나누는 것이 아닌 0과 1 사이의 값이다
#  [0.]
#  [0.]
#  [0.]]
# [0 0 0 0]
# [0 0 0 0]

# results
# [0.9799739122390747, 0.9649122953414917]
# [0 0 0 0]
# [0 0 0 0]