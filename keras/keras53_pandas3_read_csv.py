import numpy as np
import pandas as pd

df=pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)
# index_col=0 -> index 는 데이터가 아니다 라고 리드할 때 명시해줘야함
# header=0 ->header 는 default 가 있다고 존재하기 때문에 header 가 없는 경우엔 데이터가 한 줄 사라질 수 있음
# default : index_col=0, header=1
# index_col=None, 0, 1
# header=None, 0, 1
print(df)