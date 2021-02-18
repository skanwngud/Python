import p11_car # p11_car.py 의 module 이름은 :  p11_car - __main__ 이 아니라 파일명이 나옴
import p12_tv # p12_tv.py 의 module 이름은 :  p12_tv

print('='*50)
print('p13_do.py 의 module 이름은 : ', __name__) # p13_do.py 의 module 이름은 :  __main__
print('='*50)

# __main__ 은 이 파일이 실행 되는 기준이므로
# P11, p12 의 __main__ 은 p13 에서 불러왔기 때문에 파일명이 기재 된다

p11_car.drive()
p12_tv.watch() # 파일 내에 있는 함수만 가져옴