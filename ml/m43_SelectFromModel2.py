# 1. 상단모델에 그리드 서치 또는 랜덤 서치로 튜닝한 모델 구성하고 최적의 R2 값과 피처임포턴스 구할 것
# 2. 위의 쓰레드 값으로 SelectFromModel 을 구해서 최적의 피처 갯수 구할 것
# 3. 위 피터 갯수로 데이터(피처)를 수정 (삭제)하여 그리드서치 또는 랜덤서치 적용하여 최적의 R2 값 구할 것

# 1번값과 2번값 비교

import numpy as np

from xgboost import XGBRegressor

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x,y=load_boston(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=23
)

print(x_train.shape) # (404, 13)

# model

parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
]

model=XGBRegressor(
    n_jobs=-1,
    )

fi=model.feature_importances_
print(fi)