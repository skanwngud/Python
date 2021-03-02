import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    return np.maximum( 2*(np.exp(x) - 1) * abs(x)/-x , x )

x=np.arange(-5, 5, 0.1)
y=elu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()