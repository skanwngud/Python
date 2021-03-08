import numpy as np

from sklearn.covariance import EllipticEnvelope

a=np.array([[1, 2, 3, 4, 6, 7, 90, 100, 200, 1000, 20000],
            [1000, 2000 ,3 , 4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])
a=np.transpose(a)

print(a.shape) # (11, 2)

outlier=EllipticEnvelope(
    contamination=.3 # 오염매개변수 .2 : (20%), .1 : (10%) - 전체 데이터를 이상치로 판단하는 비율 (default : .1)
)

outlier.fit(a)

print(outlier.predict(a))

# [ 1  1  1  1  1  1  1  1  1 -1 -1] .2
# [ 1  1  1  1  1  1  1  1  1 -1  1] .1
# [ 1  1  1  1  1  1  1  1  1 -1  1] .3

# [ 1  1  1  1  1  1  1  1 -1 -1 -1] (2차원 배열) .3 : 열 기준이므로 하나라도 이상치가 나오면 그 열 위치를 표기를한다

# EllipticEnvelope 를 sklearn 공식 문서로 확인할 것

