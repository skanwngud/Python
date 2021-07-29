# categorical classification

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier # 분류모델
from sklearn.linear_model import LogisticRegression # 분류모델
from sklearn.ensemble import RandomForestClassifier # 분류모델
from sklearn.tree import DecisionTreeClassifier # 분류모델

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

# ss=StandardScaler()
# ss.fit(x)
# x=ss.transform(x)

mms=MinMaxScaler()
mms.fit(x)
x=mms.transform(x)
'''
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)
'''

''' - 머신러닝은 OneHotEncoding 할 필요가 없음
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)
# to_categorical 을 사용하여 데이터 * 3 으로 수가 증가함
# x에 대한 전처리는 반드시해야하지만 y에 대한 전처리는 다중분류인 경우에만 함
'''
# print(x.shape) # (150, 4)
# print(y.shape) # (150, )
# print(y_train)
# print(y_train.shape) # (96, 3)
# print(y_test.shape) # (54, 3)


# 2. Modeling
# input1=Input(shape=4)
# dense1=Dense(150, activation='relu')(input1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# output1=Dense(3, activation='softmax')(dense1)
# # softmax - 다중분류에서 사용하는 activation / 분류하려는 숫자의 갯수만큼 노드를 잡는다
# # 노드의 갯수만큼 값이 분리가 되며 각 값의 합은 1이며 가장 높은 값을 갖는게 선택이 됨
# model=Model(inputs=input1, outputs=output1)

# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=RandomForestClassifier()
# model=DecisionTreeClassifier()
model=LogisticRegression()

# 3. Compile, fitting
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류 - categorical_crossentropy, 이진분류 - binary_crossentropy
# model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val))

model.fit(x,y)

# 4. Evaluate, predict
# loss=model.evaluate(x_test, y_test)

results=model.score(x,y)
y_pred=model.predict(x) # predict 는 그대로 사용함

print('acc : ', results)
print('accuracy_score : ', accuracy_score(y, y_pred))

# results - model.compile(metrics=['acc']) 를 지정해줘야함
# [0.11219839006662369, 0.9666666388511658]
# [[1.6016660e-10 6.5629654e-05 9.9993432e-01]
#  [1.0000000e+00 6.0144376e-11 3.4863581e-26]
#  [9.9662763e-01 3.3723419e-03 5.3760271e-11]
#  [6.4497421e-05 2.8200355e-01 7.1793193e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]

# results (LinearSVC) - model.score 에서 자동으로 accuracy 를 넣어줌 
# acc :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# acc(StandardScaler) :  0.9466666666666667
# accuracy_score(StandardScaler) :  0.9466666666666667
# acc(MinMaxScaler) :  0.9466666666666667
# accuracy_score(MinMaxScaler) :  0.9466666666666667

# results (SVC)
# acc :  0.9733333333333334
# accuracy_score :  0.9733333333333334
# acc(StandardScaler) :  0.9733333333333334
# accuracy_score(StandardScaler) :  0.9733333333333334
# acc(MinMaxScaler) :  0.98
# accuracy_score(MinMaxScaler) :  0.98

# results (KNeighbors)
# acc :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# acc(StandardScaler) :  0.9533333333333334
# accuracy_score(StandarScaler) :  0.9533333333333334
# acc(MinMaxScaler) :  0.96
# accuracy_score(MinMaxScaler) :  0.96

# results (LogisticRegressor)
# acc :  0.9733333333333334
# accuracy_score :  0.9733333333333334
# acc(StandardScaler) :  0.9733333333333334
# accuracy_score(StandardScaler) :  0.9733333333333334
# acc(MinMaxScaler) :  0.94
# accuracy_score(MinMaxScaler) :  0.94

# results (RandomForest)
# acc :  1.0
# accuracy_score :  1.0
# Scaler 에 따른 변화 없음

# results (DecisionTree)
# acc :  1.0
# accuracy_score :  1.0
# Scaler 에 따른 변화 없음

# Tensorflow
# acc : 0.9666666388511658