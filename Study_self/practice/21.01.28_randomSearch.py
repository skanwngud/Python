# 0. import library
import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. data
from sklearn.datasets import load_wine

# dataset=pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) # csv 파일로 열기

dataset=load_wine()
x=dataset.data
y=dataset.target

parameters=[
        {'n_estimators':[50, 100, 150, 200]},
        {'criterion':['gini', 'entropy']},
        {'max_depth':[2, 4, 6, 8, 10]}, {'max_features' : ["auto", "sqrt", "log2"]},{'max_leaf_nodes':[2,4,6,8]},
        {'min_samples_split':[1, 2, 4, 8, 10]}, {'min_samples_leaf':[1, 2, 3, 4, 5]}, {'min_weight_fraction_leaf':[1, 2, 3, 6, 9]}
]

kf=KFold(n_splits=10, shuffle=True, random_state=23)

for train_index, val_index in kf.split(x):
    x_train, x_val=x[train_index], x[val_index]
    y_train, y_val=y[train_index], y[val_index]

print(x_train.shape) # (161, 13)
print(y_train.shape) # (161, )

# 2. model
model=RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kf)

# 3. fitting
model.fit(x_train, y_train)

# 4. predict
y_pred=model.predict(x_val)

print('best_estimator : ', model.best_estimator_)
print('score : ', model.score(x_val, y_val))
print('acc : ', accuracy_score(y_val, y_pred))
print('cross_val : ', cross_val_score(RandomForestClassifier(), x_train, y_train, cv=kf))

# best_estimator :  RandomForestClassifier(min_samples_split=10)
# score :  0.9411764705882353
# acc :  0.9411764705882353
# cross_val :  [0.94117647 1.         1.         1.         1.         0.9375        1.         1.         0.9375     1.        ]