# 매개변수
# 매개변수는 뒤에서부터 default 값을 지정해줘야한다
def addition(a = 1 , b = 3):  # a = 1, b = 3 은 default
    return a + b

print(addition())
print(addition(3))  # 매개변수의 제일 뒤부터 채워짐, 즉 a = default, b = 3
print(addition(b = 1))

# 수가 정의 되지 않은 매개변수
# 리스트 형태로 변수가 들어온다
def addition2(*args):
    for i in args:
        print(i)

addition2(1, 2, 3, 4, 5)

# 수가 정의 되지 않은 dictionary 매개 변수
def addition3(**kwargs):
    return kwargs

print(addition3(k1 = 1, k2 = 2, k3 = 3))