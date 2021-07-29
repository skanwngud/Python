import numpy as np

from sklearn.datasets import load_diabetes

from sklearn.decomposition import PCA

datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x.shape, y.shape) # (442, 10) (442,)

pca=PCA(n_components=9)

x2=pca.fit_transform(x) # fit_transfor : fit, transform 을 같이 사용함

print(x2)
print(x2.shape) # (442, 7)

pca_EVR=pca.explained_variance_ratio_ # column 의 변화율
print(pca_EVR) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
               # 각 컬럼간의 중요도
print(sum(pca_EVR))
# 압축율 7 : 0.9479436357350414
# 압축율 8 : 0.9913119559917797
# 압축율 9 : 0.9991439470098977

# 압축을 9 로 했을 때의 압축율이 0.999... 이므로 하나 정도는 빼도 됨
# 압축을 7 로 했을 때의 결과치가 9로 했을 때의 결과치와 비슷하다면 7을 사용 가능
# 통상적으로 0.95 정도면 성능에 큰 차이가 없다고 함