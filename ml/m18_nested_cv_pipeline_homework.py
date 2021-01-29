# Model = RandomForest
# Data = Diabetes
# # pipeline 엮어서 25번 돌리기 (m17 참고)

# train, val, test / kfold = 5

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.datasets import load_diabetes

datasets=load_diabetes()
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
    {'randomforestregressor__n_estimators' : [100, 200]},
    {'randomforestregressor__max_depth' : [6, 8, 10, 12]},
    {'randomforestregressor__min_samples_leaf' : [3, 5, 7, 10]},
    {'randomforestregressor__min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'randomforestregressor__max_features' : ["auto", "sqrt", "log2"]},
    {'randomforestregressor__min_samples_split' : [2, 3, 5, 10]},
    {'randomforestregressor__n_jobs' : [-1]}
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
    pipe=Pipeline([('mms', i), ('model', RandomForestRegressor())])
    for j in search:
        model=j(pipe, parameters1, cv=kf)
        score=cross_val_score(model, x_test, y_test, cv=kf)
        print(str(i), str(j) + ' : ', score, 'pipeline')

print('\nmake_pipeline')
for i in scaler:
    pipe=make_pipeline(i, RandomForestRegressor())
    for j in search:
        model=j(pipe, parameters2, cv=kf)
        score=cross_val_score(model, x_test, y_test, cv=kf)
        print(str(i), str(j) + ' : ', score, 'make_pipeline')

time_end=datetime.datetime.now()
print('spent_time : ', time_end-time_start)

# results
# Pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [ 0.33204135 -0.03308911  0.39051167 -0.17434823  0.4636863 ] pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [ 0.3697837   0.01320308  0.4413831  -0.16826961  0.54415436] pipeline
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [ 0.35508771  0.14079811  0.36392974 -0.16676115  0.41017232] pipeline
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [ 0.34053115  0.07093988  0.41436949 -0.13339998  0.41305716] pipeline

# make_pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [ 0.34036122  0.16366929  0.34072394 -0.16450071  0.44626065] make_pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [ 0.36115207  0.15026095  0.33134236 -0.04112897  0.60446247] make_pipeline
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> :  [ 0.35479398  0.18414575  0.3624108  -0.11590134  0.41636799] make_pipeline
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> :  [ 0.36490813  0.12314691  0.34268217 -0.13297158  0.57006853] make_pipeline
# spent_time :  0:03:51.158468