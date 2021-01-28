parameters=[
    {'n_estimators' : [100, 200]},
    {'criterion' : ["mse", "mae"]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'min_weight_fraction_leaf' : [2,3,4,5,6]},
    {'max_features' : ["auto", "sqrt", "log2"]},
    {'max_leaf_nodes': [10, 20, 30]},
    {'n_jobs' : [-1]}
]
import warnings
warnings.filterwarnings('ignore')
import datetime

from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score
datasets=load_diabetes()
x=datasets.data
y=datasets.target

time_start=datetime.datetime.now()

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=23)

kf=KFold(n_splits=5, shuffle=True, random_state=23)

model=GridSearchCV(RandomForestRegressor(), parameters, cv=kf)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)

time_end=datetime.datetime.now()

print(model.best_estimator_)
print(model.score(x_test, y_test))
print(r2_score(y_test, y_pred))
print(time_end-time_start)

# RandomForestRegressor(min_samples_leaf=5)
# 0.41301553970930605
# 0.41301553970930605

# RandomForestRegressor(min_samples_leaf=10)
# 0.4142709469304381
# 0.4142709469304381

# 0:00:19.945239