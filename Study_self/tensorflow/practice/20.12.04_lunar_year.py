a=2001

if a%100==0 and a%400!=0:
    print(str(a)+'년은 평년입니다')
elif a%4==0:
    print(str(a)+'년은 윤년입니다')
else:
    print(str(a)+'년은 평년입니다')