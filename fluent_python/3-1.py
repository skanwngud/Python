tt = (1, 2, (30, 40))
print(hash(tt))

tf = (1, 2, frozenset([30, 40]))
print(hash(tf))

# tl = (1, 2, [30, 40])
# print(hash(tl))

# __hash__ 메소드와 __eq__ 메소드를 가질 수 있으면 해시가능하다.
# 불변하는 객체들은 보통 해시가 가능한데 튜플의 경우 가변적인 객체를 참조 할 수 있으므로 무조건 해시가 가능하진 않다.

# 딕셔너리 생성법
a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('three', 3), ('one', 1)])
e = dict({'three': 3, 'one': 1, 'two': 2})

print(a)
print(b)
print(c)
print(d)
print(e)

print(a == b == c == d == e)
print(a is b is c is d is e)  # == 는 값, is 는 id (참조 객체)
