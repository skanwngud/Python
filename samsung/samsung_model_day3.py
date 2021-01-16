import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error, r2_score

datasets=np.load('../data/npy/samsung_day_3.npz')

x_train=datasets['x_train']
x_test=datasets['x_test']
x_val=datasets['x_val']
x_pred=datasets['x_pred']

x_1_train=datasets['x_1_train']
x_1_test=datasets['x_1_test']
x_1_val=datasets['x_1_val']
x_1_pred=datasets['x_1_pred']

y_train=datasets['y_train']
y_test=datasets['y_test']
y_val=datasets['y_val']

model=load_model('../data/modelcheckpoint/Samsung_day_3_1671099.3750.hdf5')

loss=model.evaluate([x_test, x_1_test], y_test)
y_pred=model.predict([x_test, x_1_test])

pred=model.predict([x_pred, x_1_pred])

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
print('Samsung_predict : ', pred)