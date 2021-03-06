# train, test 분리 후 train 만 validation 하지말고
# kfold 한 후에 train_test_split 사용

# 1. kfold -> 2. train_test_split

import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

datasets=load_iris()
x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=23)


model=[LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier()]

for i in model:
    score=cross_val_score(i, x, y, cv=kf)
    print('score : ', score, ' - '+str(i))

# x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=23)

# print('\n')

# for i in model:
#     score1=cross_val_score(i, x_train, y_train, cv=kf)
#     print('score : ', score1, ' - '+str(i)+' - train_test_split')

# for i in model:
#     score2=cross_val_score(i, [x_train, x_val], [y_train, y_val], cv=kf)
#     print('score : ', score2, ' - '+str(i)+' - train')
    

for train_index, test_index in kf.split(x):
    x_train, x_test=x[train_index], x[test_index]
    y_train, y_test=y[train_index], y[test_index]

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=33)

print(x.shape)
print(y.shape)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

for i in model:
    print('score : ', cross_val_score(i, x_train, y_train), ' - ' + str(i) +' _ train')

# results
# score :  [0.96666667 0.96666667 0.9        0.83333333 0.93333333]  - LinearSVC()
# score :  [0.96666667 0.96666667 0.96666667 0.96666667 0.96666667]  - SVC()
# score :  [1.         0.96666667 0.93333333 0.93333333 0.96666667]  - KNeighborsClassifier()
# score :  [1.         0.96666667 0.96666667 0.93333333 0.96666667]  - LogisticRegression()
# score :  [0.96666667 0.93333333 0.96666667 0.93333333 0.96666667]  - RandomForestClassifier()
# score :  [1.         0.96666667 0.9        0.93333333 0.9       ]  - DecisionTreeClassifier()


# score :  [0.95833333 0.875      1.         0.95833333 1.        ]  - LinearSVC() - train_test_split
# score :  [0.95833333 0.95833333 0.95833333 1.         0.95833333]  - SVC() - train_test_split
# score :  [1.         0.95833333 0.95833333 0.95833333 1.        ]  - KNeighborsClassifier() - train_test_split
# score :  [0.95833333 0.95833333 0.95833333 0.95833333 1.        ]  - LogisticRegression() - train_test_split
# score :  [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667]  - RandomForestClassifier() - train_test_split
# score :  [0.95833333 0.95833333 0.95833333 0.95833333 0.875     ]  - DecisionTreeClassifier() - train_test_split


# score :  [0.95       0.94736842 0.94736842 1.         0.94736842]  - LinearSVC() _ train
# score :  [0.95       1.         1.         0.94736842 0.89473684]  - SVC() _ train
# score :  [0.95       0.94736842 1.         1.         0.94736842]  - KNeighborsClassifier() _ train
# score :  [0.95       1.         1.         1.         0.89473684]  - LogisticRegression() _ train
# score :  [0.9        1.         1.         1.         0.89473684]  - RandomForestClassifier() _ train
# score :  [0.9        0.94736842 1.         1.         0.94736842]  - DecisionTreeClassifier() _ train