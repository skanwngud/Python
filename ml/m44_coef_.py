import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

x=[-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y=[-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

print(x,'\n', y)

plt.plot(x,y)
# plt.show()

df=pd.DataFrame({'X' : x, 'Y' : y}) # dictionary 인 (10, 2) 형태의 DataFrame 으로 바꿈

print(df)
print(df.shape) # (10, 2)

x_train=df.loc[:, 'X'] # X 의 컬럼을 x_train 으로 받음
y_train=df.loc[:, 'Y'] # Y 의 컬럼을 y_train 으로 받음

print(x_train.shape, y_train.shape) # (10, ) (10, )
print(type(x_train)) # series (Scalar 만 들어가면 Series, Vector 부터 DataFrame)

x_train=x_train.values.reshape(len(x_train), 1) # x_train 값을 reshape
print(x_train.shape, y_train.shape) # (10, 1) (10, )

model=LinearRegression()
model.fit(
    x_train, y_train
)

score=model.score(
    x_train, y_train
)
print('score : ', score)
print('coef_(기울기, weights) : ', model.coef_) # [1. ] [2. ]
print('intercept(절편, bias) : ', model.intercept_) # 1.0 3.0