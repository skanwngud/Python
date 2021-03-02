import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5, 5, 0.1) # -5 부터 5까지 0.1씩
y=sigmoid(x) # 0과 1 사이로 수렴이 됨

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()