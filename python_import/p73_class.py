class Person: # class 안에는 변수, 함수도 넣을 수 있다
    def __init__(self, name, age, address): # self : 자기 자신을 받음, __init__ 초기화 (initialize)
        self.name=name # self. 가 반드시 명시 되어야함
        self.age=age
        self.address=address

    def greeting(self): # 클래스 안에 들어가는 함수 등은 self 가 들어가야한다
        print('안녕하세요. 저의 이름은 {0}이고, 나이는 {1} 입니다.'.format(self.name, self.age)) # {0} 부분은 .format(self.name) 을 받는다
