# 집합 기본 메소드

s = {1, 2, 3}
print(s)

s.add(4)  # 집합에 요소 추가
print(s)

ss = s.copy()  # 얕은 복사
print(ss)

s.discard(1)  # 요소 제거
print(s)
s.discard(5)
print(s)

print(s.pop())

s.remove(3)
print(s)

s.clear()  # 집합의 요소 전부 제거
print(s)
print(ss)
