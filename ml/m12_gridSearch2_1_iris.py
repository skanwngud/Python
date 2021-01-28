# RandomForestClassifier

parameters=[
    {'n_estimators' : [100, 200, 300]},
    {'max_depth' : [6, 8, 10, 12, 14]},
    {'min_samples_leaf' : [3, 5, 7, 9, 10]},
    {'min_samples_split' : [2, 3, 5, 9, 10]}, # 예측값을 찾기 위해 샘플을 자른 횟수
    {'n_jobs' : [-1, 2, 4]} # 훈련할 때 돌아갈 코어갯수 (-1 이면 전부 사용)
]

import warnings
warnings.filterwarnings('ignore')

import datetime

time_start=datetime.datetime.now()

from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score

datasets=load_iris()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

model=GridSearchCV(RandomForestClassifier(), parameters, cv=kf)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)

time_end=datetime.datetime.now()


print(model.best_estimator_)
print(model.score(x_test, y_test))
print(accuracy_score(y_test, y_pred))
print(time_end-time_start)

# RandomForestClassifier(min_samples_leaf=5)
# 1.0
# 1.0

# RandomForestClassifier(min_samples_leaf=5)
# 1.0
# 1.0

# 0:00:18.677975