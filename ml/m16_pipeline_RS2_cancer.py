# import libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# data
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target

parameters=[
    {'model__n_estimators' : [100, 200]},
    {'model__max_depth' : [6, 8, 10, 12]},
    {'model__min_samples_leaf' : [3, 5, 7, 10]},
    {'model__min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'model__max_features' : ["auto", "sqrt", "log2"]},
    {'model__min_samples_split' : [2, 3, 5, 10]},
    {'model__n_jobs' : [-1]}
]

kfold=KFold(n_splits=5, shuffle=True)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# model
from sklearn.pipeline import Pipeline, make_pipeline

model=Pipeline([('mms', MinMaxScaler()), ('model', SVC())])
model=make_pipeline(MinMaxScaler(), SVC())

scaler=[MinMaxScaler(), StandardScaler()]
search=[RandomizedSearchCV, GridSearchCV]

for i in scaler:
    pipe=Pipeline([('mms', i), ('model', RandomForestClassifier())])
    for j in search:
        model=j(pipe, parameters, cv=kfold)
        model.fit(x_train, y_train)

        results=model.score(x_test, y_test)

        print(str(i), str(j)+' : '+str(results))

# results
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> : 0.9649122807017544
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> : 0.9649122807017544
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> : 0.9736842105263158
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> : 0.9736842105263158