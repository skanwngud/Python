import p21_car # p21_car 에서 if 문으로 __name__ 을 __main__ 과 일치시킬 때만 출력을 시키라 그래서 임포트 자체만으로는 아무것도 출력 되지 않는다
import p22_tv

print('='*50)
p21_car.drive()
p22_tv.watch()