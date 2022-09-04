# mappingproxy 예제

from types import MappingProxyType

d = {1: 'A'}
d_proxy = MappingProxyType(d)
print(d_proxy)
print(d_proxy[1])

# d_proxy[2] = 'X'  # MaapingProxy 를 직접 변경하려고 하면 TypeError 발생

d[2] = 'B'
print(d_proxy)
print(d_proxy[2])

