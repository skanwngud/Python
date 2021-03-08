# 이상치 처리
# 1. 이상치를 결측치(nan) 으로 바꾼 후 보간법 사용
# 2. 0 처리
# 3. 4. 5.....

import numpy as np
import matplotlib.pyplot as plt

a=np.array([1, 2, 3, 4, 6, 7, 90, 100, 200, -200]) # 평균 : 1383.45454... 중위 : 6~7 사이 / 1사분위 : 3~4사이 / 3사분위 : 90~100 사이...

def outliers(data_out):
    quartile_1, q2, quartile3=np.percentile(data_out, [25, 50, 75]) # precentile : 데이터의 사분위
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile3)
    iqr=quartile3 - quartile_1
    lower_bound=quartile_1-(iqr*1.5)
    upper_bound=quartile3+(iqr*1.5) # 1.5 가 수치적으로 가장 많이 쓰이는 방식
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc=outliers(a)
print("이상치의 위치 : ", outlier_loc)

plt.boxplot(a, sym='bo')
plt.show()

'''
1사분위 :  3.25
q2 :  6.5
3사분위 :  97.5
이상치의 위치 :  (array([8, 9], dtype=int64),) : a=np.array([1, 2, 3, 4, 6, 7, 90, 100, 5000, 10000])

1사분위 :  3.25
q2 :  6.5
3사분위 :  97.5
이상치의 위치 :  (array([5, 9], dtype=int64),) : a=np.array([1, 2, 3, 4, 6, 10000,7, 90, 100, 5000])
'''

# 현재 배운 MinMax, Standard 와 같은 scaler 인데, 위의 데이터 경우 제대로 처리가 되지 않음
# np.percentile 알아볼 것