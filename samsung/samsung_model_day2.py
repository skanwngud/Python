import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score, mean_squared_error

# datasets=np.load('../data/npy/samsung_data_2.npz')
datasets=np.load('../data/npy/samsung_data_2(whole_data).npz')

x_train=datasets['x_train']
x_test=datasets['x_test']
x_val=datasets['x_val']
x_pred=datasets['x_pred']

y_train=datasets['y_train']
y_test=datasets['y_test']
y_val=datasets['y_val']

model=load_model('../data/modelcheckpoint/samsung_day2_21133234176.0000.hdf5')

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

pred=model.predict(x_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
print('samsung_predict : ', pred)

# results - samsung_day2_750079.1250
# loss :  833800.875
# RMSE :  913.1271
# R2 :  0.9908402596203343
# samsung_predict :  [[89473.36]]