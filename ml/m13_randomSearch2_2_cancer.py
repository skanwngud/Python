parameters=[
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_weight_fraction_leaf' : [3, 5, 7, 10]},
    {'max_features' : ["auto", "sqrt", "log2"]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1]}
]

import warnings
warnings.filterwarnings('ignore')

import datetime

time_start=datetime.datetime.now()

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

model=RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kf)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)
time_end=datetime.datetime.now()

print(model.best_estimator_)
print(model.score(x_test, y_test))
print(accuracy_score(y_test, y_pred))
print(time_end-time_start)


# RandomForestClassifier(max_depth=8)
# 0.9649122807017544
# 0.9649122807017544

# RandomForestClassifier(max_features='sqrt')
# 0.9649122807017544
# 0.9649122807017544

# 0:00:08.456219