import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # --Regressor 회귀모델
from sklearn.linear_model import LogisticRegression # 분류 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# mms=MinMaxScaler()
# mms.fit(x)
# x=mms.transform(x)

ss=StandardScaler()
ss.fit(x)
x=ss.transform(x)

# model=LinearSVC()
# model=SVC()
# model=KNeighborsClassifier()
# model=RandomForestClassifier()
# model=DecisionTreeClassifier()
model=LogisticRegression()

model.fit(x_train,y_train)

loss=model.score(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(accuracy_score(y_test, y_pred))

# linear SVC
# 0.9210526315789473
# 0.9210526315789473

# SVC
# 0.956140350877193
# 0.956140350877193

# KN
# 0.956140350877193
# 0.956140350877193

# Random
# 0.956140350877193
# 0.956140350877193

# Decision
# 0.9298245614035088
# 0.9298245614035088

# Logistic
# 0.9649122807017544
# 0.9649122807017544

# Tensorflow
# acc : 0.9736841917037964