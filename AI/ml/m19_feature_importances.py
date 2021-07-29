# 0. import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

# 1. data
dataset=load_iris()

x_train, x_test, y_train, y_test=train_test_split(
        dataset.data, dataset.target, train_size=0.8, random_state=23
)

# 2. model
model=DecisionTreeClassifier(max_depth=4)

# 3. fitting
model.fit(x_train, y_train)

# 4. score, predict
acc=model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# results
# [0.02419697 0.         0.05742738 0.91837565] - feature(==column, 열, 특성 등)의 중요도. 즉 4번째의 중요도가 가장 높다
# acc :  0.9666666666666667