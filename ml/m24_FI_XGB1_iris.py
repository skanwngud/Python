# Gradient Boost

# 0. import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

from sklearn.datasets import load_iris

# 1. data
dataset=load_iris()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

start_time=datetime.datetime.now()

# x_train=x_train[:, 1:]
# x_test=x_test[:, 1:]

# 2. model
# model=GradientBoostingClassifier()
model=XGBClassifier(n_jobs=1)

# 3. fitting
model.fit(x_train, y_train)

# 4. score, predict
acc=model.score(x_test, y_test)
fi=model.feature_importances_

def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

print(cut_columns(model.feature_importances_,dataset.feature_names,8))

print(model.feature_importances_)
print('acc : ', acc)

end_time=datetime.datetime.now()
spent_time=end_time-start_time
print(spent_time)

# def plot_feature_importances(model):
#     n_features=dataset.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#         align='center') # bar 형태의 그래프
#     plt.yticks(np.arange(n_features), dataset.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances(model)
# plt.show()

# results
# [0.09649293 0.02822648 0.4012841  0.47399649]
# acc :  0.9666666666666667

# [0.20461069 0.35917997 0.43620934]
# acc :  1.0

# [0.02004825 0.63242566 0.34752609]
# acc :  0.9666666666666667

# n_jobs=-1
# [0.00840395 0.01940055 0.79458624 0.17760932]
# acc :  0.966666666
# 0:00:00.098738

# n_jons=8
# [0.00840395 0.01940055 0.79458624 0.17760932]
# acc :  0.9666666666666667
# 0:00:00.092723

# n_jobs=4
# [0.00840395 0.01940055 0.79458624 0.17760932]
# acc :  0.9666666666666667
# 0:00:00.087736

# n_jobs=1
# [0.00840395 0.01940055 0.79458624 0.17760932]
# acc :  0.9666666666666667
# 0:00:00.088273