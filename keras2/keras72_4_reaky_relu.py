import numpy as np
import matplotlib.pyplot as plt

def LeakyReLU(x):
    return np.maximum(0.01*x,x)

x=np.arange(-5, 5, 0.1)
y=LeakyReLU(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()