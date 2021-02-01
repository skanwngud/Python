# model=RF

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

datasets=load_diabetes()
x=datasets.data
y=datasets.target

for i in range(1, 11):
    pca=PCA(n_components=i)

    x2=pca.fit_transform(x)

    # pca_EVR=pca.explained_variance_ratio_

    model=RandomForestRegressor()

    # pca=PCA()
    # pca.fit(x)
    cumsum=np.cumsum(pca.explained_variance_ratio_) # 주어진 축에 대한 누적합
    # print('cumsum :', cumsum)

    model.fit(x2, y)
    r2=model.score(x2, y)

    # cumsum
    # cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
    #  0.94794364 0.99131196 0.99914395 1.        ]

    # column 이 1일 때 0.4, 2일 때 0.55 ... 10개 전부 썼을 때 1.

    d=np.argmax(cumsum>=0.95)+1
    # print('cumsum >=0.95', cumsum>=0.95) # 0.95 이상인 값들을 True, 미만 값을 False 로 표기
    # print('d : ', d) # 0.95 이상인 값을 말해줌 (출력값이 8 이므로 n_components=8 이상을 줘야함)
    print(str(i) + '번째 r2 : ', r2)

# plt.plot(cumsum)
# plt.grid()
# plt.show()

# results
# 1번째 r2 :  0.8634022975493223
# 2번째 r2 :  0.8867096382336791
# 3번째 r2 :  0.8960447511762103
# 4번째 r2 :  0.9229151116581481
# 5번째 r2 :  0.9237631585723226
# 6번째 r2 :  0.921984299560908
# 7번째 r2 :  0.9270412518150748
# 8번째 r2 :  0.9228900071289854
# 9번째 r2 :  0.9195742222204245
# 10번째 r2 :  0.9228641257234734