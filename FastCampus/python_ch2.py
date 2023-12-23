# 자료형의 복사
# 상수형의 경우 a, b 가 같은 값을 바라보다가 b 의 값이 변화하면 b 의 메모리가 새로운 변수를 저장하게 됨
a = 3
b = a
b = b + 1

print(a)
print(b)

# mutable 변수의 경우 a, b 는 서로 같은 값을 저장하고 있다가 b 의 값이 변화하면 새로운 변수를 생성하는 것이 아닌 기존의 값을 수정한다
# 동일한 메모리의 값을 서로 바라보고 있으므로 b 의 값이 변화하면 a 의 값도 변화한다
a = [1, 2, 3, 4]
b = a
b[1] = 1

print(a)
print(b)

# list.copy() 로 일반 리스트를 복사하면 깊은 복사
a = [1, 2, 3, 4]
b = a.copy()
b[1] = 1

print(a)
print(b)

# list.copy() 로 이중 리스트를 복사하면 얕은 복사
a = [1, 2, 3, 4, [5, 6, 7]]
b = a.copy()
b[4][2] = 1

print(a)
print(b)

# 이중 리스트 이상의 객체를 복사하기 위해선 deepcoy(list) 사용
from copy import deepcopy
a = [1, 2, 3, 4, [5, 6, 7]]
b = deepcopy(a)
b[4][2] = 1

print(a)
print(b)