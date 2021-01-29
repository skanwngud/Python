# import libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# data
dataset=load_diabetes()
x=dataset.data
y=dataset.target

parameters1=[
    {'model__n_estimators' : [100, 200]},
    {'model__max_depth' : [6, 8, 10, 12]},
    {'model__min_samples_leaf' : [3, 5, 7, 10]},
    {'model__min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'model__max_features' : ["auto", "sqrt", "log2"]},
    {'model__min_samples_split' : [2, 3, 5, 10]},
    {'model__n_jobs' : [-1]}
]

parameters2=[
    {'randomforestregressor__n_estimators' : [100, 200]},
    {'randomforestregressor__max_depth' : [6, 8, 10, 12]},
    {'randomforestregressor__min_samples_leaf' : [3, 5, 7, 10]},
    {'randomforestregressor__min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'randomforestregressor__max_features' : ["auto", "sqrt", "log2"]},
    {'randomforestregressor__min_samples_split' : [2, 3, 5, 10]},
    {'randomforestregressor__n_jobs' : [-1]}
]

kfold=KFold(n_splits=5, shuffle=True)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# model
from sklearn.pipeline import Pipeline, make_pipeline

model=Pipeline([('mms', MinMaxScaler()), ('model', SVC())])
model=make_pipeline(MinMaxScaler(), SVC())

scaler=[MinMaxScaler(), StandardScaler()]
search=[RandomizedSearchCV, GridSearchCV]


print('Pipeline')
for i in scaler:
    pipe=Pipeline([('mms', i), ('model', RandomForestRegressor())])
    for j in search:
        model=j(pipe, parameters1, cv=kfold)
        model.fit(x_train, y_train)

        results=model.score(x_test, y_test)

        print(str([i, j])+' : '+str(results))

print('\nmake_pipeline')
for i in scaler:
    pipe=make_pipeline(i, RandomForestRegressor())
    for j in search:
        model=j(pipe, parameters2, cv=kfold)
        model.fit(x_train, y_train)

        results=model.score(x_test, y_test)

        print(str([i, j])+' : '+str(results))
        

# results
# Pipeline
# [MinMaxScaler(), <class 'sklearn.model_selection._search.RandomizedSearchCV'>] : 0.41271745243689517
# [MinMaxScaler(), <class 'sklearn.model_selection._search.GridSearchCV'>] : 0.44592592047128465
# [StandardScaler(), <class 'sklearn.model_selection._search.RandomizedSearchCV'>] : 0.4141963542701439
# [StandardScaler(), <class 'sklearn.model_selection._search.GridSearchCV'>] : 0.4133194752833015
# make_pipeline
# [MinMaxScaler(), <class 'sklearn.model_selection._search.RandomizedSearchCV'>] : 0.42151160293395074
# [MinMaxScaler(), <class 'sklearn.model_selection._search.GridSearchCV'>] : 0.4195435694048566
# [StandardScaler(), <class 'sklearn.model_selection._search.RandomizedSearchCV'>] : 0.4251905438782623
# [StandardScaler(), <class 'sklearn.model_selection._search.GridSearchCV'>] : 0.4061541255605223