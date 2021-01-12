import numpy as np

from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

# 1. boston
boston_datasets=load_boston()
boston_x=boston_datasets.data
boston_y=boston_datasets.target

np.save('../data/npy/boston_x.npy', arr=boston_x)
np.save('../data/npy/boston_y.npy', arr=boston_y)

# 2. diabets
diabets_datasets=load_diabetes()
diabets_x=diabets_datasets['data']
diabets_y=diabets_datasets['target']

np.save('../data/npy/diabets_x.npy', arr=diabets_x)
np.save('../data/npy/diabets_y.npy', arr=diabets_y)

# 3. cancer
cancer_datasets=load_breast_cancer()
cancer_x=cancer_datasets.data
cancer_y=cancer_datasets.target

np.save('../data/npy/cancer_x.npy', arr=cancer_x)
np.save('../data/npy/cancer_y.npy', arr=cancer_y)

# 4. wine
wine_datasets=load_wine()
wine_x=wine_datasets.data
wine_y=wine_datasets.target

np.save('../data/npy/wine_x.npy', arr=wine_x)
np.save('../data/npy/wine_y.npy', arr=wine_y)

# 5. mnist
(m_x_train, m_y_train), (m_x_test, m_y_test)=mnist.load_data()

np.save('../data/npy/mnist_x_train.npy', arr=m_x_train)
np.save('../data/npy/mnist_x_test.npy', arr=m_x_test)
np.save('../data/npy/mnist_y_train.npy', arr=m_y_train)
np.save('../data/npy/mnist_y_test.npy', arr=m_y_test)

# 6. fashion_mnist
(fm_x_train, fm_y_train), (fm_x_test, fm_y_test)=fashion_mnist.load_data()

np.save('../data/npy/fashion_mnist_x_train.npy', arr=fm_x_train)
np.save('../data/npy/fashion_mnist_x_test.npy', arr=fm_x_test)
np.save('../data/npy/fashion_mnist_y_train.npy', arr=fm_y_train)
np.save('../data/npy/fashion_mnist_y_test.npy', arr=fm_y_test)

# 7. cifar10
(cf_x_train, cf_y_train), (cf_x_test, cf_y_test)=cifar10.load_data()

np.save('../data/npy/cifar10_x_train.npy', arr=cf_x_train)
np.save('../data/npy/cifar10_x_test.npy', arr=cf_x_test)
np.save('../data/npy/cifar10_y_train.npy', arr=cf_y_train)
np.save('../data/npy/cifar10_y_test.npy', arr=cf_y_test)

# 8. cifar100
(cf100_x_train, cf100_y_train), (cf100_x_test, cf100_y_test)=cifar100.load_data()

np.save('../data/npy/cifar100_x_train.npy', arr=cf100_x_train)
np.save('../data/npy/cifar100_x_test.npy', arr=cf100_x_test)
np.save('../data/npy/cifar100_y_train.npy', arr=cf100_y_train)
np.save('../data/npy/cifar100_y_test.npy', arr=cf100_y_test)