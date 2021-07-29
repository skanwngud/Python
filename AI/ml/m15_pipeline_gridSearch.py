# import libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# data
dataset=load_iris()
x=dataset.data
y=dataset.target

kfold=KFold(n_splits=5, shuffle=True)

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

# parameters=[
#     {'model__C':[1, 10, 100, 1000], 'model__kernel':['model__linear']},
#     {'model__C':[1, 10, 100], 'model__kernel':['rbf'], 'model__gamma':[0.001, 0.0001]},
#     {'model__C':[1, 10, 100, 1000], 'model__kernel':['sigmoid'], 'model__gamma':[0.001, 0.0001]}
# ]

parameters=[
    {'mod__C':[1, 10, 100, 1000], 'mod__kernel':['mod__linear']},
    {'mod__C':[1, 10, 100], 'mod__kernel':['rbf'], 'mod__gamma':[0.001, 0.0001]},
    {'mod__C':[1, 10, 100, 1000], 'mod__kernel':['sigmoid'], 'mod__gamma':[0.001, 0.0001]}
]

# pipeline 으로 작성할 때 model 부분으로 정의한 '문자열(소문자)__' 부분으로 작성

# mms=MinMaxScaler()
# mms.fit(x_train)
# x_train=mms.transform(x_train)
# x_test=mms.transform(x_test)

# model
from sklearn.pipeline import Pipeline, make_pipeline # 둘 다 동일 / 전처리와 모델을 합침

pipe=Pipeline([('mms', MinMaxScaler()), ('model', SVC())]) # mms 와 SVC 를 합침
pipe=make_pipeline(MinMaxScaler(), SVC())
# 위의 MinMaxScaler 를 따로 정의하지 않아도 Pipeline 에서 정의해주면 자동으로 합쳐준다

model=GridSearchCV(pipe, parameters, cv=kfold) # make_Pipeline 은 돌아가지 않음

scaler=[MinMaxScaler(), StandardScaler()]

for i in scaler:
    # pipe=Pipeline([('mms', i), ('model', SVC())])
    pipe=Pipeline([('mms', i), ('mod', SVC())])
    model=GridSearchCV(pipe, parameters, cv=kfold)
    model.fit(x_train, y_train)

    results=model.score(x_test, y_test)

    print(str(i)+' : '+str(results))

# only pipeline
# MinMaxScaler() : 0.9666666666666667
# StandardScaler() : 1.0

# gridSearch
# MinMaxScaler() : 0.9666666666666667
# StandardScaler() : 0.9666666666666667