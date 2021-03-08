import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

dataset=load_boston()
x=dataset.data
y=dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506, )
print('='*20)
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names) # column name

# x=x/711. # -- > 각 컬럼의 최대 최소값이 다른데 전체 데이터셋의 최대값으로 나누기만 했음
# 데이터의 형변환 때문에 (float 형)

print(np.max(x[0])) # 396.9
# x 의 첫 번째 컬럼(CRIM)의 최대값



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

scaler=RobustScaler() # 중위값을 이용한 스케일링
scaler.fit(x)
x=scaler.transform(x)

print(np.min(x)) # -18.76100251828754
print(np.max(x)) # 24.678376790228196
print(np.max(x[0])) # 1.44

# scaler=StandardScaler() # 표준편차를 이용한 스케일링
# scaler.fit(x)
# x=scaler.transform(x)

# print(np.max(x)) # 9.933930601860268
# print(np.min(x)) # -3.9071933049810337
# print(np.max(x[0])) # 0.44105193260704206
'''
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

x_test, x_pred, x_val 전부 이미 위에서 w 값이 나왔기 때문에 transform 만 하면 된다

오히려 범위를 벗어난 부분이 생기면 훈련을 더 잘 할 수 있게 된다 (과적합 방지)
'''

'''
print(np.max(x), np.min(x)) # 711.0 0.0 -> 1.0 0.0
print(np.max(x[0])) # 0.9999999999999999

# 전처리 전
# [13.635822296142578, 2.712141990661621]
# 3.6926712892525257
# 0.8368587933642436

# 전처리 후 - x=x/711.
# [12.10439682006836, 2.588151454925537]
# 3.479137453964179
# 0.855180999735316 - 노드 35개

# 전처리 후 - MinMaxScaler
# [8.861637115478516, 2.0854930877685547]
# 2.9768507285717583
# 0.8939778794202871
'''