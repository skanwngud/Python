# 0. import libraries

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

# 1. data

datasets=load_iris()
x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=22)

for train_index, test_index in kf.split(x):
    x_train, x_test=x[train_index], x[test_index]
    y_train, y_test=y[train_index], y[test_index]

x_train, x_val, y_train, y_val=train_test_split(x_train, y_train,
            train_size=0.8, random_state=33)

print(x_train.shape) # (96, 4)
print(x_val.shape) # (24, 4)
print(x_test.shape) # (30, 4)

parameters1=[
        {}
]
parameters2=[
        {}
]
parameters3=[
        {}
]

model=[LinearSVC(), SVC(), RandomForestClassifier(),
        KNeighborsClassifier(), DecisionTreeClassifier()]

for i in model:
    