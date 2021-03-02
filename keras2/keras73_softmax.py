import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x)) # np.exp 찾아볼 것

x=np.arange(1, 5) # 1 부터 5까지 1씩
y=softmax(x) # 전부 합치면 1

ratio=y
labels=y

plt.pie(
    ratio,
    labels=labels,
    shadow=True,
    startangle=90
)
plt.show()