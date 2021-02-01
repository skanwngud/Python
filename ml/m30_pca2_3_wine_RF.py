import numpy as np

from sklearn.datasets import load_wine

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

datasets=load_wine()
x=datasets.data
y=datasets.target

print(x.shape)

for i in range(1, 14):
    pca=PCA(n_components=i)
    x2=pca.fit_transform(x)

    model=RandomForestClassifier()
    model.fit(x2,y)

    acc=model.score(x2, y)

    print(str(i) + ' 번째 acc : ', acc)

# results
# 1 번째 acc :  1.0
# 2 번째 acc :  0.9943820224719101
# 3 번째 acc :  1.0
# 4 번째 acc :  1.0
# 5 번째 acc :  1.0
# 6 번째 acc :  1.0
# 7 번째 acc :  1.0
# 8 번째 acc :  1.0
# 9 번째 acc :  1.0
# 10 번째 acc :  1.0
# 11 번째 acc :  1.0
# 12 번째 acc :  1.0
# 13 번째 acc :  1.0