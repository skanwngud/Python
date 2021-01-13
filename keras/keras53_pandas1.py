import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

dataset=load_iris()

# print(dataset.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(dataset.values())
# print(dataset.target_names) # ['setosa' 'versicolor' 'virginica']

x=dataset.data
# x=dataset['data']
y=dataset.target
# y=dataset['target']

print(x)
print(y)
print(x.shape) # (150, 4)
print(y.shape) # (150, )
print(type(x), type(y)) # <class 'numpy.ndarray'>

df=pd.DataFrame(x, columns=dataset.feature_names)
# df=pd.DataFrame(x, columns=dataset['feature_names'])
print(df)
print(df.shape) # list 에는 shape 를 출력할 수 없음 (그래서 np.array[] 로 표현)
print(df.columns)
print(df.index) # RangeIndex(start=0, stop=150, step=1)
print(df.head()) # == df[:5]
print(df.tail()) # == df[-5:]
print(df.info())
print(df.describe())

df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # column name 갱신
print(df.columns)
print(df.info())
print(df.describe())

print(df['sepal_length'])
df['Target']=dataset.target # y 값 (target) 붙이기

print(df.head())
print(df.shape) # (150, 5)
print(df.columns)
print(df.tail())
print(df.info())

print(df.isnull()) # 결측치
print(df.isnull().sum()) # 결측치의 합 (갯수 출력)

print(df.describe())
print(df['Target'].value_counts())

# 상관계수
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 도수분포도 # histogram / model.fit 에서 반환 되는 hist 는 history
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length', data=df)
plt.title('tepal_length')

plt.subplot(2,2,4)
plt.hist(x='petal_width', data=df)
plt.title('tepal_length')

plt.show()


'''
# df.columns
index
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) - header, column
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]

index, header = No data

# df.index
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64 - 결측치 없음
dtypes: float64(4)
memory usage: 4.8 KB
None

# df.describe
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000 - 갯수
mean            5.843333          3.057333           3.758000          1.199333 - 평균
std             0.828066          0.435866           1.765298          0.762238 - 표준편차
min             4.300000          2.000000           1.000000          0.100000 - 최소값
25%             5.100000          2.800000           1.600000          0.300000 - 상위 25% 값 (상위 값)
50%             5.800000          3.000000           4.350000          1.300000 - 가운데 값(중위 값)
75%             6.400000          3.300000           5.100000          1.800000 - 상위 75% 값 (하위 값)
max             7.900000          4.400000           6.900000          2.500000 - 최대값

# df.isnull().sum
[150 rows x 5 columns]
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
Target          0
dtype: int64

# 상관 계수
              sepal_length  sepal_width  petal_length  petal_width    Target
sepal_length      1.000000    -0.117570      0.871754     0.817941  0.782561
sepal_width      -0.117570     1.000000     -0.428440    -0.366126 -0.426658 - 상관 계수가 제일 적은 데이터이므로 조정 (제거) 가능
petal_length      0.871754    -0.428440      1.000000     0.962865  0.949035
petal_width       0.817941    -0.366126      0.962865     1.000000  0.956547
Target            0.782561    -0.426658      0.949035     0.956547  1.000000
'''