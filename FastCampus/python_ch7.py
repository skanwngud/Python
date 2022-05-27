# 상속
class Plus:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def plus(self):
        return self.x + self.y
    
class Minus(Plus):
    def minus(self):
        return self.x - self.y
    
    
a = Plus(5, 4)
print(a.plus())

b = Minus(5, 4)
print(b.minus())


# 추상 클래스와 isinstance
# 추상 클래스는 보통 자신이 클래스를 생성할 때 반드시 넣어줘야하는 메소드를 생성할 때 사용한다 (까먹을 수 있으므로)
from abc import *
class Abstract(metaclass=ABCMeta):
    @abstractmethod  # 데코레이터 함수로 인해 해당 메소드를 하위 클래스에 강제한다
    def method(self):  # 추상 메소드
        pass
    
    class test(Abstract):  # 상위의 Abstract 라는 클래스를 상속 받았다
        def method(self):  # 데코레이터 함수인 method 라는 추상 메소드를 강제로 넣어야한다
            pass

