# Gradient Boost

# 0. import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.datasets import load_wine

# 1. data
dataset=load_wine()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

# x_train=x_train[:, 1:]
# x_test=x_test[:, 1:]

statr_time=datetime.datetime.now()

# 2. model
# model=GradientBoostingClassifier()
jobs=[-1, 8, 4, 1]
for i in jobs:
    model=XGBClassifier(n_jobs=i, use_label_encoder=False, eval_mtrics='logloss')

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

    # print(cut_columns(model.feature_importances_,dataset.feature_names,8))

    # print(model.feature_importances_)
    print(str(i) + ' acc : ', acc)

    end_time=datetime.datetime.now()
    spent_time=end_time-statr_time

    print(str(i) + ' spent_time : ', spent_time)

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
# acc :  1.0
# 0:00:00.090759

# n_jobs=8
# acc :  1.0
# 0:00:00.080784

# n_jobs=4
# 4 acc :  1.0
# 4 spent_time :  0:00:00.147574

# n_jobs=1
# 1 acc :  1.0
# 1 spent_time :  0:00:00.190460