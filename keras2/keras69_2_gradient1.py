import numpy as np
import matplotlib.pyplot as plt

f=lambda x:x**2-4*x+6 # 2차 함수
x=np.linspace(-1, 6, 100) # -1 부터 6까지 100개가 들어감
y=f(x)

# 시각화
plt.plot(x, y, 'k-') # 'k-' 는 그래프의 색 (blac'k')
plt.plot(2,2, 'sk') # 그래프 상 최적의 w 값 / 'sk'는 해당 결과값에 점을 찍어줌
plt.grid()

plt.xlabel('x')
plt.ylabel('y')
plt.show()