import numpy as np

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.utils import to_categorical

a=np.array([3,6,5,4,2])

# a=a.reshape(1, -1)
# enc=OneHotEncoder()
# enc.fit(a)
# a=enc.transform(a).toarray()

# print(a)
# print(a.shape)

# # [[0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 1.]
# #  [0. 0. 0. 1. 0.]
# #  [0. 0. 1. 0. 0.]
# #  [1. 0. 0. 0. 0.]]
# # (5, 5)

a=to_categorical(a)

print(a)
print(a.shape)

# [[0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0.]]
# (5, 7)