# categorical classification

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, KFold, cross_val_score # cross_val_score - 교차검증값
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

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

kfold=KFold(n_splits=5, shuffle=True) # n_splits= 몇 등분할 지


# 2. modeling
model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=RandomForestClassifier()
# model=DecisionTreeClassifier()
# model=LogisticRegression()

score=cross_val_score(model, x_train,y_train, cv=kfold) # cv = Cross Validation : 교차 검증을 무엇으로 할 지

print('score : ', score)
# score :  [1.         0.96666667 0.96666667 0.96666667 0.86666667] - model.fit, model.score 까지 포함 됨
# score :  [0.875      0.95833333 1.         1.         0.95833333] - train_test_split 후 결과값