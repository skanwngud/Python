import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel

from tensorflow.keras.utils import to_categorical

x,y=load_wine(return_X_y=True) # 데이터 로드

kf=KFold(
    n_splits=5,
    shuffle=True,
    random_state=23
)

for train_index, test_index in kf.split(x,y):
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# print(x_train.shape) # (143, 13)
# print(x_test.shape) # (35, 13)

# print(y_train.shape) # (143, 3)

# print(y_train)

