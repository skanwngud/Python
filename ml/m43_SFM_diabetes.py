# 0.5 이상

# 1. 상단모델에 그리드 서치 또는 랜덤 서치로 튜닝한 모델 구성하고 최적의 R2 값과 피처임포턴스 구할 것
# 2. 위의 쓰레드 값으로 SelectFromModel 을 구해서 최적의 피처 갯수 구할 것
# 3. 위 피터 갯수로 데이터(피처)를 수정 (삭제)하여 그리드서치 또는 랜덤서치 적용하여 최적의 R2 값 구할 것

# 1번값과 2번값 비교

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor, XGBClassifier

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x,y=load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=23
)

print(x_train.shape) # (353, 10)

# model

parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]}
]

model=RandomizedSearchCV(
    XGBRegressor(
        n_jobs=-1
    ),
    parameters
)

model.fit(
    x_train, y_train
)

score=model.score(
    x_test, y_test
)

y_pred=model.predict(
    x_test
)

thresholds=np.sort(model.best_estimator_.feature_importances_)

print('feature_importance : ', thresholds)
print('best_params : ', model.best_params_)
print('r2 : ', r2_score(y_test, y_pred))
print('acc : ', score)

# feature_importance :  [0.04160768 0.04494249 0.05286829 0.05361563 0.06416546 0.06444992
#  0.08838359 0.11148928 0.19030985 0.2881678 ]
# best_params :  {'n_estimators': 90, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
# r2 :  0.4009702020101834
# acc :  0.4009702020101834


for thresh in thresholds: # 9개
    selection=SelectFromModel(
        model.best_estimator_, threshold=thresh, prefit=True
    )
    select_x_train=selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape)

    selection_model=RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters)
    selection_model.fit(select_x_train, y_train)

    select_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2 : %.2f%%'%(thresh, select_x_train.shape[1], score*100))

# n=9