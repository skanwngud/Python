# tuple unpacking

print(divmod(20, 8))
a = (20, 8)
print(divmod(*a))
q, r = divmod(*a)
print(q, r)

import os

_, filename = os.path.split("../fluent_python/2-8.py")
print(filename)

a, b, *rest = range(5)
print(a, b, rest)

a, b, *rest = range(3)
print(a, b, rest)

a, b, *rest = range(2)
print(a, b, rest)