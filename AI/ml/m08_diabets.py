import numpy as np

from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes

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
print('sacler : ', str(scaler))
print('model_score : ', loss)
print('r2_score : ', r2_score(y_test, y_pred))

# results -  LinearRegression()
# model_score :  0.7451430642919576
# r2_score :  0.7451430642919576
# sacler :  MinMaxScaler()
# model_score :  -27345.471337230716
# r2_score :  -27345.471337230716
# sacler :  StandardScaler()
# model_score :  -3147.844945651135
# r2_score :  -3147.844945651135

# results -  KNeighborsRegressor()
# model_score :  0.5222138859051638
# r2_score :  0.5222138859051638
# sacler :  MinMaxScaler()
# model_score :  -1.0150460890927282
# r2_score :  -1.0150460890927282
# sacler :  StandardScaler()
# model_score :  -0.8356283198549166
# r2_score :  -0.8356283198549166

# results -  DecisionTreeRegressor()
# model_score :  0.7569767032511389
# r2_score :  0.7569767032511389
# sacler :  MinMaxScaler()
# model_score :  -1.004335573205422
# r2_score :  -1.004335573205422
# sacler :  StandardScaler()
# model_score :  -0.5772727305694589
# r2_score :  -0.5772727305694589

# results -  RandomForestRegressor()
# model_score :  0.8313291932013951
# r2_score :  0.8313291932013951
# sacler :  MinMaxScaler()
# model_score :  -0.036132511910646814
# r2_score :  -0.036132511910646814
# sacler :  StandardScaler()
# model_score :  -0.0021651692378894527
# r2_score :  -0.0021651692378894527

# Tensorflow
# r2 :  0.5114865328386683