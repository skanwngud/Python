import numpy as np

x=np.array([1.,2.,3.,4.,5.])

print(np.where(x>=3)) # (array([2,3,4]), dtype=int64) - 특정 값의 위치와 타입을 반환
print(x[np.where(x>=3)]) # [3. 4. 5.] - 특정 값을 반환
print(np.where(x>=3, 3, 0)) # [0 0 3 3 3] - 특정 값을 명시 된 값으로 변환 후 반환
                            # x 가 3 이상인 경우엔 3, 아닌 경우엔 0으로 변환 후 반환