from sklearn.datasets import load_iris

import numpy as np

datasets=load_iris()

# print(datasets) # data : array, target : array, frame : None, target_names : array[setosa, versicolor, virginica] ==> key : value

print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x_data=datasets.data
y_data=datasets.target
# x_data=datasets['data']
# y_data=datasets['target']

# print(x_data)
# print(y_data)

print(datasets.frame)
print(datasets.target_names)
print(datasets['DESCR'])
print(datasets['feature_names'])
print(datasets.filename)

print(type(x_data), type(y_data)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# np.save('../data/npy/iris_x.npy', arr=x_data)
# np.save('../data/npy/data/iris_y.npy', arr=y_data)

# load_iris 하는 데이터는 key : value 로 이루어져 있다. (딕셔너리 자료형)