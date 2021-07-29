# RandomForest
# data = wine

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.datasets import load_wine
datasets=load_wine()
x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=23)

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
    {'randomforestclassifier__n_estimators' : [100, 200]},
    {'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
    {'randomforestclassifier__min_samples_leaf' : [3, 5, 7, 10]},
    {'randomforestclassifier__min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'randomforestclassifier__max_features' : ["auto", "sqrt", "log2"]},
    {'randomforestclassifier__min_samples_split' : [2, 3, 5, 10]},
    {'randomforestclassifier__n_jobs' : [-1]}
]

for train_index, test_index in kf.split(x, y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]


for train_index, val_index in kf.split(x_train, y_train):
    x_train=x[train_index]
    x_val=x[val_index]
    y_train=y[train_index]
    y_val=y[val_index]

print(x_train.shape) # (284, 10)
print(x_test.shape) # (88, 10)
print(x_val.shape) # (70, 10)
print(y_train.shape) # (284, )
print(y_test.shape) # (88, )
print(y_val.shape) # (70, )


scaler=[MinMaxScaler(), StandardScaler()]
search=[GridSearchCV, RandomizedSearchCV]

import datetime

time_start=datetime.datetime.now()

print('Pipeline')
for i in scaler:
    pipe=Pipeline([('mms', i), ('model', RandomForestClassifier())])
    for j in search:
        model=j(pipe, parameters1, cv=kf)
        score=cross_val_score(model, x_test, y_test, cv=kf)
        print(str(i), str(j) + ' : ', score, 'pipeline')

print('\nmake_pipeline')
for i in scaler:
    pipe=make_pipeline(i, RandomForestClassifier())
    for j in search:
        model=j(pipe, parameters2, cv=kf)
        score=cross_val_score(model, x_test, y_test, cv=kf)
        print(str(i), str(j) + ' : ', score, 'make_pipeline')

time_end=datetime.datetime.now()
print('spent_time : ', time_end-time_start)

# results
# Pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [0.85714286 1.         1.         1.         1.        ] pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [1. 1. 1. 1. 1.] pipeline
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [0.85714286 1.         1.         1.         1.        ] pipeline
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [0.85714286 1.         1.         0.85714286 1.        ] pipeline

# make_pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [0.85714286 1.         1.         1.         1.        ] make_pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [1. 1. 1. 1. 1.] make_pipeline
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [0.85714286 1.         1.         0.85714286 1.        ] make_pipeline
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [1. 1. 1. 1. 1.] make_pipeline
# spent_time :  0:04:04.866360