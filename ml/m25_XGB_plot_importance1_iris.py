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
from xgboost import XGBClassifier, plot_importance

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

# def cut_columns(feature_importances,columns,number):
#     temp = []
#     for i in feature_importances:
#         temp.append(i)
#     temp.sort()
#     temp=temp[:number]
#     result = []
#     for j in temp:
#         index = feature_importances.tolist().index(j)
#         result.append(columns[index])
#     return result

# print(cut_columns(model.feature_importances_,dataset.feature_names,8))

print(model.feature_importances_)
print('acc : ', acc)

end_time=datetime.datetime.now()
spent_time=end_time-start_time
print(spent_time)

'''
def plot_feature_importances(model):
    n_features=dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center') # bar 형태의 그래프
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()
'''

plot_importance(model) # F score 가 기준 / 상관관계가 높은 순으로 정렬
plt.show()

# 과제 F_score 정리