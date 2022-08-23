# 슬라이싱

a = """
name..........country..........coordinate
Seoul         Korea            (123, 1234)
Tokyo         Japan            (234, 2345)
"""

name = slice(0, 6)
county = slice(14, 22)
city_items = a.split('\n')[2:]

for item in city_items:
    print(item[name], item[county])

# 다차원 슬라이싱
a = list(range(10))
a = [0,...]
print(a)  # 파이썬 내부 함수의 리스트는 1차원 리스트만 사용가능하며, 다차원 배열을 사용하는 numpy 등에서 활용 가능하다

# 슬라이싱에 할당
l = list(range(10))
print(l)
l[2:5] = [20, 30]
print(l)
del l[5:7]
print(l)
l[3::2] = [11, 22]
print(l)
l[2:5] = [100]  # 할당문의 대상이 슬라이스인 경우 하나의 값을 넣어줘야 할 때에도 반복가능한 객체를 넣어줘야한다, = 100 으로 하면 TypeError
print(l)
