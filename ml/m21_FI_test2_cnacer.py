# 시각화

# 0. import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

# 1. data
dataset=load_breast_cancer()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

x_train=x_train[:, [4,13,20,21,22,26,27]]
x_test=x_test[:, [4,13,20,21,22,26,27]]

# 2. model
model=DecisionTreeClassifier(max_depth=4)

# 3. fitting
model.fit(x_train, y_train)

# 4. score, predict
acc=model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

def plot_feature_importances(model):
    n_features=dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center') # bar 형태의 그래프
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances(model)
plt.show()

# results
# [0.         0.04310936 0.         0.         0.00850457 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.04247645 0.         0.         0.         0.
#  0.         0.         0.1145186  0.02062783 0.01256385 0.
#  0.         0.         0.00975456 0.73247174 0.01597303 0.        ]
# acc :  0.9385964912280702

# [0.01505309 0.05190096 0.1145186  0.06373719 0.01256385 0.00975456
#  0.73247174]
# acc :  0.9385964912280702