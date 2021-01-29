# 0. import library

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score

# 1. Data
dataset=load_iris()
x=dataset.data
y=dataset.target

kfold=KFold(n_splits=5, shuffle=True) # n_splits= 몇 등분할 지

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

parameters=[
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}
]

# 2. modeling
model=GridSearchCV(SVC(), parameters, cv=kfold) # 위의 파라미터를 종류별로 돌게 된다

# 3. fitting
model.fit(x_train, y_train)

# 4. evaluate, predidct
print('최적의 매개 변수 : ', model.best_estimator_) # best_estimators 90번 돌 동안 어느 값이 가장 좋은지를 출력함

y_pred=model.predict(x_test)
print('최종 정답률 : ', accuracy_score(y_test, y_pred))
print('모델 스코어 : ', model.score(x_test, y_test))

# 최적의 매개 변수 :  SVC(C=1, kernel='linear')
# 최종 정답률 :  1.0
# 모델 스코어 :  1.0