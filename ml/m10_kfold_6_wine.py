
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine

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
dataset=load_wine()
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

# score :  [0.82758621 0.79310345 0.89285714 0.64285714 0.64285714] -LinearSVC()
# score :  [0.68965517 0.65517241 0.75       0.71428571 0.57142857] -SVC()
# score :  [0.79310345 0.72413793 0.78571429 0.78571429 0.78571429] -KNeighborsClassifier()
# score :  [1.         0.93103448 1.         0.96428571 1.        ] -RandomForestClassifier()
# score :  [0.93103448 0.86206897 0.96428571 0.89285714 0.89285714] -DecisionTreeClassifier()
# score :  [1.         0.96551724 0.92857143 0.96428571 0.89285714] -LogisticRegression()