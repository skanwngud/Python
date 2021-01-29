
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split, KFold, cross_val_score # cross_val_score - 교차검증값
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_diabetes()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

kfold=KFold(n_splits=5, shuffle=True) # n_splits= 몇 등분할 지


# 2. modeling
model=[LinearSVC(), SVC(), KNeighborsRegressor(), RandomForestRegressor(), DecisionTreeRegressor()]

for i in model:
    score=cross_val_score(i, x_train,y_train, cv=kfold)
    print('score : ', score, '-'+str(i))

# score :  [0.         0.         0.         0.01428571 0.        ] -LinearSVC()
# score :  [0.01408451 0.         0.         0.         0.        ] -SVC()
# score :  [0.22985747 0.36646139 0.48536663 0.29656839 0.32902582] -KNeighborsRegressor()
# score :  [0.45099222 0.49495802 0.28668954 0.45257916 0.47615638] -RandomForestRegressor()
# score :  [-0.15024623  0.17410174  0.06880575  0.15754608 -0.26206992] -DecisionTreeRegressor()