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

from sklearn.datasets import load_wine

# 1. data
dataset=load_wine()

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

# results
# ['proanthocyanins', 'nonflavanoid_phenols', 'total_phenols', 'alcalinity_of_ash', 'magnesium', 'malic_acid', 'alcohol', 'od280/od315_of_diluted_wines']
# [1.69424277e-02 1.51723549e-02 3.63831082e-02 5.27243940e-03
#  9.29196517e-03 9.51138348e-04 2.96620727e-01 7.88986063e-04
#  1.56526154e-05 2.90907113e-01 3.01352743e-02 2.89537368e-02
#  2.68565077e-01]
# acc :  1.0