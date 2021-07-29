
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split, KFold, cross_val_score # cross_val_score - 교차검증값
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_boston()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

kfold=KFold(n_splits=5, shuffle=True) # n_splits= 몇 등분할 지


# 2. modeling
model=[LinearSVC(), SVC(), KNeighborsRegressor(), RandomForestRegressor(), DecisionTreeRegressor()]
model1=LinearSVC()
model2=SVC()
model3=KNeighborsRegressor()
model4=RandomForestRegressor()
model5=DecisionTreeRegressor()

for i in model:
    score=cross_val_score(i, x_train,y_train, cv=kfold)
    print('score : ', score, '-'+str(i))

# score :  [nan nan nan nan nan] -LinearSVC() - 분류모델이지만 회귀 데이터이므로 nan 값이 나옴
# score :  [nan nan nan nan nan] -SVC() - 분류모델이지만 회귀 데이터이므로 nan 값이 나옴
# score :  [0.4517506  0.30114225 0.58984048 0.37730775 0.51495069] -KNeighborsRegressor()
# score :  [0.85471295 0.91729148 0.8777191  0.88662878 0.75582482] -RandomForestRegressor()
# score :  [0.6138081  0.80208306 0.78872756 0.58525477 0.72352577] -DecisionTreeRegressor()