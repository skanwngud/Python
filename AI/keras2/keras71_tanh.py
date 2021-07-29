# -1, 1 사이 수렴

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5, 5, 0.1)
y=np.tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()