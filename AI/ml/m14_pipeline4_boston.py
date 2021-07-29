
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split, KFold, cross_val_score # cross_val_score - 교차검증값
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_boston()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

scaler=[MinMaxScaler(), StandardScaler()]

for i in scaler:
    model=make_pipeline(i, RandomForestRegressor())
    model.fit(x_train, y_train)

    result=model.score(x_test, y_test)

    print(str(i)+' : '+str(result))

# MinMaxScaler() : 0.8255007445701881
# StandardScaler() : 0.8323781044516111