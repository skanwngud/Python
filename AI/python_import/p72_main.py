import p71_byunsu as p71

print(p71.aaa)
print(p71.square(10))

print('='*50)

from p71_byunsu import aaa, square # 변수도 import 가능

aaa=3

print(aaa)
print(p71.square(10))