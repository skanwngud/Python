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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, \
    RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer

scaler=PowerTransformer(
    method='yeo-johnson'
)

scaler.fit(x)
x=scaler.transform(x)

print(np.min(x)) # -4.478632778203448
print(np.max(x)) # 3.6683978597124267
print(np.max(x[0])) # 1.6052699155846277

# scaler=MaxAbsScaler()
# scaler.fit(x)
# x=scaler.transform(x)

# print(np.min(x)) # 0.0
# print(np.max(x)) # 1.0
# print(np.max(x[0])) # 1.0

# scaler=QuantileTransformer() # 1000개의 분위수를 줌 (데이터가 1000개 이하면 애매~함)
# scaler=QuantileTransformer(
#     output_distribution='normal' # 정규분포 (default : 'uniform' - 균등분포)
# )
# scaler.fit(x)
# x=scaler.transform(x)

# print(np.min(x)) # 0.0
# print(np.max(x)) # 1.0
# print(np.max(x[0])) # 1.0

# scaler=RobustScaler() # 중위값을 이용한 스케일링
# scaler.fit(x)
# x=scaler.transform(x)

# print(np.min(x)) # -18.76100251828754
# print(np.max(x)) # 24.678376790228196
# print(np.max(x[0])) # 1.44

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

print(np.max(x), np.min(x)) # 711.0 0.0 -> 1.0 0.0
print(np.max(x[0])) # 0.9999999999999999
'''