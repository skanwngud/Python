import numpy as np

from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel # feature_importance 와 관련
from sklearn.metrics import r2_score, accuracy_score

x,y=load_boston(return_X_y=True) # return_X_y : x,y 로 바로 분리 됨

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=23
)

model=XGBRegressor(
    n_jobs=8
)

model.fit(
    x_train, y_train
)

score=model.score(
    x_test, y_test
)

print('R2 : ', score)

thresholds=np.sort(model.feature_importances_) # 오름차순으로 정렬
print(thresholds)

'''
R2 :  0.8536327501944545
[0.00246722 0.00315469 0.00744116 0.01017815 0.01372482 0.02105495
 0.02376373 0.03370734 0.03572574 0.03997704 0.05112228 0.19373403
 0.56394887]
'''

for thresh in thresholds:
    selection=SelectFromModel(
        model, threshold=thresh, prefit=True
    )
    select_x_train=selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(select_x_train.shape)

    selection_model=XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test=selection.transform(x_test)
    y_predict=selection_model.predict(select_x_test)

    score=r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2 : %.2f%%'%(thresh, select_x_train.shape[1], score*100))

# for 문 돌리기 전에 모델을 먼저 돌려 feature_importances 를 추출해야한다.
# (단, 위의 feature_importances 가 신뢰 할 수 있다는 가정하에)

print('가중치 : ', model.coef_)
print('바이어스 : ', model.intercept_)
# AttributeError: Coefficients are not defined for Booster type None : Booster 타입에선 coef_ 가 정의 되지 않는다
# Booster 모델은 y=ax+b 의 형태로 학습을 하는 게 아님 (coef_ 가 없을 뿐 다른 용어로써 존재한다)

'''
(404, 13)
Thresh=0.002, n=13, R2 : 85.36%
(404, 12)
Thresh=0.003, n=12, R2 : 85.25%
(404, 11)
Thresh=0.007, n=11, R2 : 83.56%
(404, 10)
Thresh=0.010, n=10, R2 : 85.35%
(404, 9)
Thresh=0.014, n=9, R2 : 83.77%
(404, 8)
Thresh=0.021, n=8, R2 : 84.01%
(404, 7)
Thresh=0.024, n=7, R2 : 82.38%
(404, 6)
Thresh=0.034, n=6, R2 : 84.36%
(404, 5)
Thresh=0.036, n=5, R2 : 79.98%
(404, 4)
Thresh=0.040, n=4, R2 : 78.25%
(404, 3)
Thresh=0.051, n=3, R2 : 71.92%
(404, 2)
Thresh=0.194, n=2, R2 : 66.59%
(404, 1)
Thresh=0.564, n=1, R2 : 43.27%
'''

# 과제 1. prefit 에 대해 알아올 것