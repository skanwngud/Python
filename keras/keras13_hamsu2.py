import numpy as np

x=np.array([range(100), range(301, 401), range(1, 101), range(501, 601), range(651, 751)])
y=np.array([range(711, 811), range(101, 201)])

x_pred2=np.array([100, 401, 101, 601, 751]) # (5,)

print(x.shape) # (5, 100)
print(y.shape) # (2, 100)
print(x_pred2.shape)

x=np.transpose(x)
y=np.transpose(y)
x_pred2=x_pred2.reshape(1, 5)

print(x.shape) # (100, 5)
print(y.shape) # (100, 2)
print(x_pred2.shape) 

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
# class name = Capital
# Ctrl + space
# Model = 함수형 모델
from tensorflow.keras.layers import Dense, Input

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.2, shuffle=True, random_state=66)

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 2)

input1=Input(shape=(5,))
aaa=Dense(5, activation='relu')(input1)
aaa=Dense(3)(aaa)
aaa=Dense(4)(aaa)
outputs=Dense(2)(aaa)
model=Model(inputs=input1, outputs=outputs)
model.summary()


# model=Sequential()
# model.add(Dense(5, input_shape=(5,), activation='relu'))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()



model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=130, batch_size=1, validation_split=0.2, verbose=1)

loss, mae=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2=r2_score(y_test, y_predict)

print('rmse : ', RMSE(y_test, y_predict))
print('loss : ', loss)
print('mae : ', mae)
print('r2 : ', r2)

y_pred2=model.predict(x_pred2)
print(y_pred2)
