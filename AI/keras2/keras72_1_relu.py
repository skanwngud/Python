import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x) # 0 보다 작을 땐 0, 0보다 클 땐 x 값 출력

x=np.arange(-5, 5, 0.1)
y=relu(x)

print(x)
print(y) # 0 초과부터 값을 출력

plt.plot(x,y)
plt.grid()
plt.show()

# 과제
# elu, selu, reaky relu
# keras72_2, 3, 4 번으로 파일을 만들 것