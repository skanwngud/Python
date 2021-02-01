parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
    'colsampe_bylevel':[0.6, 0.7, 0.9]}
] # XGB 의 성능에 가장 영향을 많이 주는 파라미터

from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

import warnings
warnings.filterwarnings(action='ignore')
import datetime


str_time=datetime.datetime.now()

# 1. data
datasets=load_iris()

x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=23)

for train_index, test_index in kf.split(x, y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

# 2. mpdel
model=GridSearchCV(XGBClassifier(n_jobs=-1,eval_metric='mlogloss'), parameters, cv=kf)

# 3. fitting
model.fit(x_train, y_train)
acc=model.score(x_test, y_test)
y_pred=model.predict(x_test)

end_time=datetime.datetime.now()

print(acc)
print(end_time-str_time)

# results
# 0.9666666666666667
# 0:00:53.253585