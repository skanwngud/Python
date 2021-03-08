# outliers1 을 행렬형태도 적용할 수 있도록 수정

import numpy as np
import matplotlib.pyplot as plt

a=np.array([[1,2,3,4,10000, 6, 7, 5000, 90, 100],
            [1000, 2000 ,3 , 4000, 5000, 6000, 7000, 8, 9000, 10000]])

a=a.transpose()
print(a.shape) # (10, 2)


def outliers(data_out, column):
    list = []
    for i in range(data_out.shape[1]): # a.shape[1] 이 행렬에서 열 값이므로 [1] 을 준다
        quartile_1, q2, quartile3=np.percentile(data_out[:,i], [25, 50, 75]) # precentile : 데이터의 사분위
        print("1사분위 : ", quartile_1)
        print("q2 : ", q2)
        print("3사분위 : ", quartile3)
        iqr=quartile3 - quartile_1
        lower_bound=quartile_1-(iqr*1.5)
        upper_bound=quartile3+(iqr*1.5) # 1.5 가 수치적으로 가장 많이 쓰이는 방식

        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound))
        list.append(a)

    return np.array(list)

outlier_loc=outliers(a, 2)
print("이상치의 위치 : ", outlier_loc)

plt.boxplot(a)
plt.show()