import numpy as np

from sklearn.datasets import load_breast_cancer, load_boston

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # --Regressor 회귀모델
# from sklearn.linear_model import LogisticRegression # 분류 모델
from sklearn.linear_model import LinearRegression # 회귀모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

dataset=load_boston()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# scaler=MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

# model=LinearRegression()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()

model.fit(x_train,y_train)

loss=model.score(x_test, y_test)
y_pred=model.predict(x_test)

# print('results - ', str(model))
print('scaler - ', str(scaler))
print('model_score : ', loss)
print('r2_score : ', r2_score(y_test, y_pred))

# results - LinearRegression()
# model_score :  0.7451430642919576
# r2_score :  0.7451430642919576
# scaler -  MinMaxScaler()
# model_score :  0.7451430642919576
# r2_score :  0.7451430642919576
# scaler -  StandardScaler()
# model_score :  0.7451430642919576
# r2_score :  0.7451430642919576

# results -  KNeighborsRegressor()
# model_score :  0.5222138859051638
# r2_score :  0.5222138859051638
# scaler -  MinMaxScaler()
# model_score :  0.5222138859051638
# r2_score :  0.5222138859051638
# scaler -  StandardScaler()
# model_score :  0.5222138859051638
# r2_score :  0.5222138859051638

# results -  DecisionTreeRegressor()
# model_score :  0.7437288333036062
# r2_score :  0.7437288333036062
# scaler -  MinMaxScaler()
# model_score :  0.755028219286347
# r2_score :  0.755028219286347
# scaler -  StandardScaler()
# model_score :  0.737393984151719
# r2_score :  0.737393984151719

# results -  RandomForestRegressor()
# model_score :  0.8396733068397507
# r2_score :  0.8396733068397507
# scaler -  MinMaxScaler()
# model_score :  -0.2957669500636808
# r2_score :  -0.2957669500636808
# scaler -  StandardScaler()
# model_score :  -0.02740354491879593
# r2_score :  -0.02740354491879593

# Tensorflow
# r2 : 0.93280876980413

