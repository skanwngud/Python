import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,\
    MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier


wine=pd.read_csv(
    'c:/data/csv/winequality-white.csv',
    index_col=None, header=0, sep=';' # index 없음, header 는 제일 첫 번째로, ; 로 분리
)

print(wine.shape); # (4898, 12)
print(wine.describe()) # mean : 평균, std : 표준편차

# numpy 화
wine_npy=wine.values

# data 분리
x=wine_npy[:, :11]
y=wine_npy[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898, )

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=23
)

ss=StandardScaler()
ss.fit(x_train)
x_train=ss.transform(x_train)
x_test=ss.transform(x_test)

print(x_train.shape, x_test.shape) # (3918, 11) (980, 11)

# modeling
# model=KNeighborsClassifier()
# model=RandomForestClassifier()
model=XGBClassifier()

model.fit(
    x_train, y_train
)

score=model.score(
    x_test, y_test
)

print('score : ', score)

# KN
# score :  0.5561224489795918

# RF
# score :  0.6836734693877551

# XGB
# score :  0.6785714285714286