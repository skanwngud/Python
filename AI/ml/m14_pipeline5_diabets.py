
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
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_diabetes()
x=dataset.data
y=dataset.target

scaler=[MinMaxScaler(), StandardScaler()]

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

for i in scaler:
    model=Pipeline([('mms', i), ('model', RandomForestRegressor())])
    model.fit(x_train, y_train)

    print(str(i)+' : '+str(model.score(x_test, y_test)))

# model=make_pipeline(MinMaxScaler(), RandomForestRegressor())


# MinMaxScaler() : 0.4981081416702561
# StandardScaler() : 0.4895865627599605