import numpy as np

from sklearn.datasets import load_boston

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

datasets=load_boston()
x=datasets.data
y=datasets.target

print(x.shape)

for i in range(1, 14):
    pca=PCA(n_components=i)
    x2=pca.fit_transform(x)

    model=RandomForestRegressor()
    model.fit(x2,y)

    r2=model.score(x2, y)

    print(str(i) + ' 번째 r2 : ', r2)

# results
# 1 번째 r2 :  0.8921082176901631
# 2 번째 r2 :  0.9120044682601777
# 3 번째 r2 :  0.9212945189807387
# 4 번째 r2 :  0.9226325761892273
# 5 번째 r2 :  0.9407350692188102
# 6 번째 r2 :  0.9612627802358963
# 7 번째 r2 :  0.9604774406676104
# 8 번째 r2 :  0.9649873967658226
# 9 번째 r2 :  0.9673832829494361
# 10 번째 r2 :  0.9683427647256949
# 11 번째 r2 :  0.9762350157442947
# 12 번째 r2 :  0.9743185573481574
# 13 번째 r2 :  0.9754203515590723