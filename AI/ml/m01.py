import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0, 10, 0.1) # 0~9 까지 0.1 단위로 표현
y=np.sin(x) # 사인함수

plt.plot(x,y)
plt.show()