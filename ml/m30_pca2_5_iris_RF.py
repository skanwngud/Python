import numpy as np

from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, KFold

datasets=load_iris()
x=datasets.data
y=datasets.target

pca=PCA(n_components=3)
x2=pca.fit_transform(x)

# x_train, x_test, y_train, y_test=train_test_split(x2, y, train_size=0.8, random_state=22)

# kf=KFold(n_splits=5, shuffle=True, random_state=23)

# for trian_index, test_index in kf.split(x2, y):
#     x_train=x2[trian_index]
#     x_test=x2[test_index]
#     y_train=y[trian_index]
#     y_test=y[test_index]

# model=RandomForestRegressor()
# model.fit(x_train, y_train)

# r2=model.score(x_test, y_test)

# print(r2)

# print(x.shape) # (150, 4)

for i in range(1, 5):
    pca=PCA(n_components=i)
    x2=pca.fit_transform(x)

    model=RandomForestRegressor()
    model.fit(x2,y)

    r2=model.score(x2, y)

    print(str(i) + ' 번째 r2 : ', r2)

# results
# 1 번째 r2 :  0.98367
# 2 번째 r2 :  0.992708
# 3 번째 r2 :  0.991554
# 4 번째 r2 :  0.991386

# n_components = 3
# train_test_split
# 0.9002164179104477

# kf.split (n_split=5)
# 0.949805900621118