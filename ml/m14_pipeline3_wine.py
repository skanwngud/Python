import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, KFold, cross_val_score # cross_val_score - 교차검증값
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC # support vector model
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier # 분류모델
from sklearn.linear_model import LogisticRegression # 분류모델
from sklearn.ensemble import RandomForestClassifier # 분류모델
from sklearn.tree import DecisionTreeClassifier # 분류모델
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. Data
dataset=load_wine()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
        x,y, train_size=0.8, random_state=33
)

scaler=[MinMaxScaler(), StandardScaler()]

for i in scaler:
    model=Pipeline([('mms', i), ('model', RandomForestClassifier())])
    model.fit(x_train, y_train)

    results=model.score(x_test, y_test)

    print(str(i)+' : '+str(results))

# MinMaxScaler() : 1.0
# StandardScaler() : 1.0