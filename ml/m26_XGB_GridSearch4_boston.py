parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
]

import warnings
warnings.filterwarnings(action='ignore')
import datetime

from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV, KFold

from xgboost import XGBRegressor

# 1. data
datasets=load_boston()
x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=22)

str_time=datetime.datetime.now()

