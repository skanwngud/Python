# 결측치 처리 (시계열에서 유리함)

from pandas import DataFrame, Series
from datetime import datetime

import numpy as np
import pandas as pd

datestrs=['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates=pd.to_datetime(datestrs)

print(dates)
print('='*50)

ts=Series([1, np.nan, np.nan, 8, 10], index=dates) # dates 에 있는 5개의 날짜와 매칭이 된다
print(ts)

'''
ts
2021-03-01     1.0
2021-03-02     NaN
2021-03-03     NaN
2021-03-04     8.0
2021-03-05    10.0
'''

ts_intp_linear=ts.interpolate() # nan 값들을 보간법으로 처리해줌 (linear 방식)
print(ts_intp_linear)

'''
ts_intp_linear
2021-03-01     1.000000
2021-03-02     3.333333
2021-03-03     5.666667
2021-03-04     8.000000
2021-03-05    10.000000
'''