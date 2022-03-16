a = None

a = "hi"

print(a)

b = None
b = [1, 2, 3]

print(b)

c = None
c = {"key": "error"}

print(c)

d = None
d = 1

print(d)

e = None
e = [None]
print(e)
print(len(e))

f = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
g = 5
h = f[g:]
print(len(f) - g)
print(len(h))

import numpy as np


for idx in range(100):
    i = np.random.randint(0, 2, size=1)
    print(i)
