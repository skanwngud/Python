# pca 통해 0.95 이상인 결과값 찾기

import numpy as np

from sklearn.decomposition import PCA

from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _)=mnist.load_data() # _ : y_train, y_test 는 쓰지 않겠다는 말

x=np.append(x_train, x_test, axis=0)

print(x.shape) # (70000, 28, 28)

x=x.reshape(-1, 28*28)

pca=PCA()
pca.fit_transform(x)

cumsum=np.cumsum(pca.explained_variance_ratio_)

print(np.argmax((cumsum>=1.0)+1))
# print(x2.shape) # (70000, 154)

# results
# 154 (0.95)
# 712 (1.0)