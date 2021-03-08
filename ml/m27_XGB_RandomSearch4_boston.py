parameters=[
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 100], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.01, 0.001],
    'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]}
]

import datetime

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, RandomizedSearchCV

from xgboost import XGBRegressor

str_time=datetime.datetime.now()

# 1. data
datasets=load_boston()
x=datasets.data
y=datasets.target

kf=KFold(n_splits=5, shuffle=True, random_state=12)

for train_index, test_index in kf.split(x, y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

# 2. model
model=RandomizedSearchCV(XGBRegressor(n_jobs=-1), parameters, cv=kf)

# 3. fitting
model.fit(x_train, y_train)
acc=model.score(x_test, y_test)

end_time=datetime.datetime.now()

print('R2 : ', acc*100)
print(end_time-str_time)

# results
# 0.8951442290912498
# 0:00:02.415538