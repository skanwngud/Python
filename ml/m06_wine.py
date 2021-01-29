import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # --Regressor 회귀모델
from sklearn.linear_model import LogisticRegression # 회귀 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. data
datasets=load_wine()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# mms=MinMaxScaler()
# mms.fit(x_train)
# x_train=mms.transform(x_train)
# x_test=mms.transform(x_test)

# ss=StandardScaler()
# ss.fit(x_train)
# x_train=ss.transform(x_train)
# x_test=ss.transform(x_test)

# 2. model
model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=LogisticRegression()
# model=DecisionTreeClassifier()
# model=RandomForestClassifier()

# 3. fitting
model.fit(x_train, y_train)

# 4. score, predict
loss=model.score(x_test, y_test)
y_pred=model.predict(x_test)

print('model : ', str(model))
print('accuracy : ', loss)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

# results - LinearSVC
# accuracy :  0.6666666666666666
# accuracy_score :  0.6666666666666666
# accuracy :  1.0 (mms)
# accuracy_score :  1.0 (mms)
# accuracy :  1.0 (ss)
# accuracy_score :  1.0 (ss)

# results - SVC
# accuracy :  0.8333333333333334
# accuracy_score :  0.8333333333333334
# accuracy :  1.0 (mms)
# accuracy_score :  1.0 (mms)
# accuracy :  1.0 (ss)
# accuracy_score :  1.0 (ss)

# results - KN
# accuracy :  0.5555555555555556
# accuracy_score :  0.5555555555555556
# accuracy :  1.0 (mms)
# accuracy_score :  1.0 (mms)
# accuracy :  1.0 (ss)
# accuracy_score :  1.0 (ss)

# results - Logistic
# accuracy :  1.0
# accuracy_score :  1.0
# accuracy :  1.0 (mms)
# accuracy_score :  1.0 (mms)
# accuracy :  1.0 (ss)
# accuracy_score :  1.0 (ss)

# results - Decision
# accuracy :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# accuracy :  0.9722222222222222 (mms)
# accuracy_score :  0.9722222222222222 (mms)
# accuracy :  0.9722222222222222 (ss)
# accuracy_score :  0.9722222222222222 (ss)

# results - RandomForest
# accuracy :  1.0
# accuracy_score :  1.0
# accuracy :  1.0 (mms)
# accuracy_score :  1.0 (mms)
# accuracy :  1.0 (ss)
# accuracy_score :  1.0 (ss)

# Tensorflow
# acc : 1.0