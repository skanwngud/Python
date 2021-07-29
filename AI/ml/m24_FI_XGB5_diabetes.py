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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from sklearn.datasets import load_diabetes

# 1. data
dataset=load_diabetes()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

# x_train=x_train[:, 1:]
# x_test=x_test[:, 1:]

# 2. model

jobs=[-1, 8, 4, 1]

start_time=datetime.datetime.now()

for i in jobs:
    # model=GradientBoostingClassifier()
    model=XGBRegressor(n_jobs=i)

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
    # print(str(i) + ' acc : ', acc)
    
    end_time=datetime.datetime.now()
    print(str(i)+ ' spent_time : ', end_time-start_time)

# n_jobs
# -1 spent_time :  0:00:00.126660
# 8 spent_time :  0:00:00.219412
# 4 spent_time :  0:00:00.310202
# 1 spent_time :  0:00:00.422898