import p01_car
import p02_tv

# 임포트할 때는 같은 폴더 내에 위치 시켜줘야한다
# 각 파일 내에 실행 명령이 있기 때문에 임포트하자마자 실행이 된다
# 실행 명령이 없으면(함수나 클래스, 변수만 있으면) 임포트만 됨

print('==========')

p01_car.drive() # import 된 함수에서 drive 라는 함수를 불러옴
p02_tv.watch() # watch 도 마찬가지