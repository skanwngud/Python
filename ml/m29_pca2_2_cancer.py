import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.decomposition import PCA

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

# pca=PCA(n_components=7)

# x2=pca.fit_transform(x)

# pca_EVR=pca.explained_variance_ratio_

pca=PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_) # 주어진 축에 대한 누적합
print('cumsum :', cumsum)

# cumsum
# cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

# column 이 1일 때 0.4, 2일 때 0.55 ... 10개 전부 썼을 때 1.

d=np.argmax(cumsum>=0.95)+1
print('cumsum >=0.95', cumsum>=0.95) # 0.95 이상인 값들을 True, 미만 값을 False 로 표기
print('d : ', d) # 0.95 이상인 값을 말해줌 (출력값이 8 이므로 n_components=8 이상을 줘야함)

plt.plot(cumsum)
plt.grid()
plt.show()