import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine=pd.read_csv(
    'c:/data/csv/winequality-white.csv',
    index_col=None, header=0, sep=';'
)

count_data=wine.groupby('quality')['quality'].count()

print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64

# 시각화
count_data.plot() 
plt.show() # 5,6,7 컬럼에 모여있다

# 대회에 참가할 때 데이터에 접근 할 권한이 있다면 y 를 조절 가능, 없다면 불가능이지만 조절하면서 결과가 어떻게 바뀌는지는 판단 가능