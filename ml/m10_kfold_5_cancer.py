# LogisticRegressor 도 사용 가능

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer

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
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

kfold=KFold(n_splits=5, shuffle=True) # n_splits= 몇 등분할 지


# 2. modeling
model=[LinearSVC(), SVC(), KNeighborsClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression()]
model1=LinearSVC()
model2=SVC()
model3=KNeighborsClassifier()
model4=RandomForestClassifier()
model5=DecisionTreeClassifier()
model6=LogisticRegression() # 2진분류

for i in model:
    score=cross_val_score(i, x_train,y_train, cv=kfold)
    print('score : ', score, '-'+str(i))

# score :  [0.96703297 0.91208791 0.84615385 0.93406593 0.96703297] -LinearSVC()
# score :  [0.95604396 0.9010989  0.93406593 0.93406593 0.87912088] -SVC()
# score :  [0.94505495 0.89010989 0.94505495 0.94505495 0.94505495] -KNeighborsClassifier()
# score :  [0.95604396 0.94505495 0.96703297 0.97802198 0.95604396] -RandomForestClassifier()
# score :  [0.94505495 0.86813187 0.87912088 0.93406593 0.96703297] -DecisionTreeClassifier()
# score :  [0.94505495 0.92307692 0.96703297 0.92307692 0.97802198] -LogisticRegression()