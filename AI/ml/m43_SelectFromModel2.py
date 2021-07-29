# 1. 상단모델에 그리드 서치 또는 랜덤 서치로 튜닝한 모델 구성하고 최적의 R2 값과 피처임포턴스 구할 것
# 2. 위의 쓰레드 값으로 SelectFromModel 을 구해서 최적의 피처 갯수 구할 것
# 3. 위 피터 갯수로 데이터(피처)를 수정 (삭제)하여 그리드서치 또는 랜덤서치 적용하여 최적의 R2 값 구할 것

# 1번값과 2번값 비교

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x,y=load_boston(return_X_y=True)

# x_train, x_test, y_train, y_test=train_test_split(
#     x,y,
#     train_size=0.8,
#     random_state=23
# )

print(x.shape) # (404, 13)

kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

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
    parameters,
    cv=kf
)

for train_index, test_index in kf.split(x, y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    
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
# thresholds=model.best_estimator_.feature_importances_

print('feature_importance : ', thresholds)
print('best_params : ', model.best_params_)
print('r2 : ', r2_score(y_test, y_pred))
print('acc : ', score)

# feature_importance :  [0.0229461  0.00231503 0.02513053 0.00437412 0.02569489 0.18420292
#  0.00834244 0.05612769 0.01053069 0.02553529 0.02401277 0.0099373 0.6008503 ]
# best_params :  {'n_estimators': 90, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 0.9, 'colsampe_bylevel': 0.7}
# r2 :  0.8311368648170403
# acc :  0.8311368648170403

for thresh in thresholds: # 10개
    selection=SelectFromModel(
        model.best_estimator_, threshold=thresh, prefit=True
    )
    select_x_train=selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape)

    selection_model=RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters, cv=kf)
    selection_model.fit(select_x_train, y_train)

    select_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2 : %.2f%%'%(thresh, select_x_train.shape[1], score*100))

# n=10 83,30%

# feature_importance :  [0.02762745 0.00770288 0.01105987 0.01125646 0.03041848 0.1997476
#  0.00849882 0.0553772  0.02132557 0.03146764 0.0536876  0.01052503 0.53130543]

# feature_importance :  [0.00655915 0.00851688 0.00922593 0.0113147  0.01202895 0.01669716
#  0.02145781 0.0239139  0.03290829 0.0360156  0.04861519 0.19811703 0.5746294]

# 0.8951442290912498
# R2 : 92.23%