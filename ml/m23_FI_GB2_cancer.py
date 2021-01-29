# Gradient Boost

# 0. import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

# 1. data
dataset=load_breast_cancer()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

# 2. model
model=GradientBoostingClassifier()

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