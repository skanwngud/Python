# 0. import libraries
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_boston, load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. data
# x,y=load_boston(return_X_y=True) # dataset 으로 x,y 잡는 것과 같은 방식

datasets=load_boston() # data load
x=datasets.data
y=datasets['target']

x_train, x_test, y_train, y_test=train_test_split(x,y, # train, test split
                        train_size=0.8, random_state=66)

# 2. model
model=XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8) # n_estimators == epochs

# 3. fitting
model.fit(x_train, y_train, verbose=1, # train, test dataset 의 loss 반환값을 출력해준다
            eval_metric=['logloss','rmse'], eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=20) # == callbacks=EarlyStopping
score=model.score(x_test, y_test) # r2_score

print('score : ', score)

# 4. predict
y_pred=model.predict(x_test)
r2=r2_score(y_test, y_pred)

print('r2 : ', r2)

results=model.evals_result() # loss 가 어떻게 변하는지에 대한 history 를 보여줌

# print('results : ', results)

epochs=results['validation_0']['logloss']
epochs1=results['validation_1']['logloss']
r_epochs=len(results['validation_0']['logloss'])

rmse=results['validation_0']['rmse']
rmse1=results['validation_1']['rmse']
x_axis=range(0, r_epochs)

fig, ax=plt.subplots()
ax.plot(x_axis, epochs, label='Train')
ax.plot(x_axis, epochs1, label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax=plt.subplots()
ax.plot(x_axis, rmse, label='Train')
ax.plot(x_axis, rmse1, label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

# results
# score :  -0.06921425433417538
# r2 :  -0.06921425433417538