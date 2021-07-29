# 0. import libraries
import numpy as np

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
            eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=10) # == callbacks=EarlyStopping
score=model.score(x_test, y_test) # r2_score

print('score : ', score)

# 4. predict
y_pred=model.predict(x_test)
r2=r2_score(y_test, y_pred)

print('r2 : ', r2)

results=model.evals_result() # loss 가 어떻게 변하는지에 대한 history 를 보여줌

# print('results : ', results)

# pickle
import pickle
pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb'))
model2=pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb'))

# joblib
import joblib
joblib.dump(model, '../data/xgb_save/m40.joblib.dat')
model3=joblib.load('../data/xgb_save/m40.joblib.dat')

# XGBoost model
model.save_model('../data/xgb_save/m41.xgb.dat')
model4=XGBRegressor()
model4.load_model('../data/xgb_save/m41.xgb.dat') # 저장 된 모델과 혼동을 주지 않기 위해 다시 모델을 정의해준다


print('='*50)
# load save data

r22=model2.score(x_test, y_test)
r23=model3.score(x_test, y_test)
r24=model4.score(x_test, y_test)

print('r22 : ', r22)
print('r23 : ', r23)
print('r24 : ', r24)

# results
# score :  0.93302293398985
# r2 :  0.93302293398985
# ==================================================
# r22 :  0.93302293398985
# r23 :  0.93302293398985
# r24 :  0.93302293398985