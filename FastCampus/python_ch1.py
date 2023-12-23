# 주석
'''
안녕하세요
Hello World!
'''

# print
print("Hello world!")
print("안녕 파이썬")

# 변수 선언
a = 2000
print(a)

a = a + 20
print(a)

a = a - 100
print(a)

# 변수 선언 2
# 변수의 값에 따라 자동으로 자료형이 정의 됨
a1 = 100  # int
a2 = 6.76  # float
a3 = "대박 사건"  # str

# 변수 선언 3
# 한 줄의 코드가 길어질 때 \ 사용
# 반대로 두 줄의 코드를 한 줄로 채우고 싶을 땐 ; 사용
print("\
    대박사건")
print("대박"); print("사건")

# 조건문 if
x = 100
if x > 0:
    print("lager than 0")
else:
    print("smaller than 0")
    
# 대입과 조건문 if
a1 = 7
a2 = 10
if a1 > a2:
    print("a1 is larger than a2")
else:
    print("a1 is smaller than a2")
    
# 기본 연산
b = 7
print(b ** 3)