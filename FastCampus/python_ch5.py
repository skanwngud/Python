# 재귀 함수
# 함수 내부에서 자기 함수를 다시 호출하는 함수
def func(count):
    if count > 0:
        print(count, "현재")
        func(count - 1)
    print("결과", count)

func(10)

# 재귀 함수의 호출 단계가 깊어질수록 메모리를 추가적으로 사용하므로 종료 조건을 분명히 해야한다
# RecursionError: maximum recursion depth exceeded in comparison 에러 발생
def func():
    print("나는 아무 생각이 없다")
    print("왜냐하면 아무 생각이 없기 떄문이다")
    func()
    
# func()

# 중첩 함수
# 함수 안의 다른 함수가 존재
def func1(a):
    def func2():
        nonlocal a  # 한 단계 위의 매개변수를 가져옴
        a += 1
        return a
    return func2()

print(func1(2))

# lambda 함수
# 재사용 되지 않고 사용 후 바로 해제 되는 함수 (이름없는 함수)
def run(func, x):
    print(func(x))

run(lambda x: x + 2, 2)

# zip, map, filter
# 반환값은 list, tuple 등으로 지정해줘야한다
# zip, 두 개의 리스트의 원소를 하나의 튜플 쌍으로 만들어줌
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
print(tuple(zip(list1, list2)))

# map, 리스트의 값들을 함수의 인자로 넣어 연산한 값들을 반환
print(list(map(lambda x: x + 100, [1, 2, 3, 4])))

# filter, 리스트의 값들을 함수의 인자로 넣어 연산한 값들 중 True 인 값만 반환
print(list(filter(lambda x: x > 10, [8, 9, 10, 11, 12, 13])))