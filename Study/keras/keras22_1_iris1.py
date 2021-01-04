# categorical classification

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
# x,y=load_iris(return_X_y=True) # 교육용에서만 사용
dataset=load_iris()
x=dataset.data
y=dataset.target

# print(x[:5])
# print(y)

## One_hot_encoding

from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical - old keras version 

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)
# to_categorical 을 사용하여 데이터 * 3 으로 수가 증가함
# x에 대한 전처리는 반드시해야하지만 y에 대한 전처리는 다중분류인 경우에만 함

# print(x.shape) # (150, 4)
# print(y.shape) # (150, )
# print(y_train)
# print(y_train.shape) # (96, 3)
# print(y_test.shape) # (54, 3)


# 2. Modeling
input1=Input(shape=4)
dense1=Dense(150, activation='relu')(input1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
output1=Dense(3, activation='softmax')(dense1)
# softmax - 다중분류에서 사용하는 activation / 분류하려는 숫자의 갯수만큼 노드를 잡는다
# 노드의 갯수만큼 값이 분리가 되며 각 값의 합은 1이며 가장 높은 값을 갖는게 선택이 됨
model=Model(inputs=input1, outputs=output1)

# 3. Compile, fitting
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류 - categorical_crossentropy, 이진분류 - binary_crossentropy
model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

# 4. Evaluate
loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x[-5:-1])

print(loss)
print(y_pred)
print(y_test[-5:-1])
print(np.argmax(y_pred, axis=-1))