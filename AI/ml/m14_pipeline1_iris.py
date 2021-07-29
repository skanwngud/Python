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

# mms=MinMaxScaler()
# mms.fit(x_train)
# x_train=mms.transform(x_train)
# x_test=mms.transform(x_test)

# model
from sklearn.pipeline import Pipeline, make_pipeline # 둘 다 동일 / 전처리와 모델을 합침

# model=Pipeline([('mms', MinMaxScaler()), ('model', SVC())]) # mms 와 SVC 를 합침
model=make_pipeline(MinMaxScaler(), SVC())
# 위의 MinMaxScaler 를 따로 정의하지 않아도 Pipeline 에서 정의해주면 자동으로 합쳐준다

scaler=[MinMaxScaler(), StandardScaler()]

for i in scaler:
    model=Pipeline([('mms', i), ('model', RandomForestClassifier())])
    model.fit(x_train, y_train)

    results=model.score(x_test, y_test)

    print(str(i)+' : '+str(results))

# MinMaxScaler() : 0.9666666666666667
# StandardScaler() : 1.0