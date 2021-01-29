# 시각화

# 0. import libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston

# 1. data
dataset=load_boston()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

# 2. model
model=DecisionTreeRegressor(max_depth=4)

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

print(dataset.feature_names)

# results
# [0.00906048 0.         0.00763128 0.         0.02236389 0.29034802
#  0.         0.06945736 0.         0.         0.00537661 0.01212754
#  0.58363482]
# acc :  0.7688217209037792