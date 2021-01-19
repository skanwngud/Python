import pandas as pd

df=pd.DataFrame([[1,2,3,4], [4,5,6,7], [7,8,9,10]],
                columns=list('abcd'), index=('가', '나', '다'))

print(df)

df2=df # pd 에서는 같은 메모리를 공유한다 ( = 만 기능하며 다른 연산자 (+, - 등) 은 새로운 df2 를 만들고 저장함)

df2['a']=100

print(df2)
print(df)

print(id(df), id(df2))

df3=df.copy() # 새로운 데이터가 생성 됨

df2['b']=333

print('='*30)
print(df)
print(df2)
print(df3)
print('='*30)

df=df+99
print(df)
print(df2)