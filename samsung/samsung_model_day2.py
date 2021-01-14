import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score, mean_squared_error

datasets=np.load('../data/npy/samsung_data_2.npz')

x_train=datasets['x_train']
x_test=datasets['x_test']
x_val=datasets['x_val']
x_pred=datasets['x_pred']

y_train=datasets['y_train']
y_test=datasets['y_test']
y_val=datasets['y_val']

model=load_model('../data/modelcheckpoint/samsung_day2_685991.1875.hdf5')

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

# results - samsung_day2_818009.4375
# loss :  883755.0
# RMSE :  940.0825
# R2 :  0.9902914861372593
# samsung_predict :  [[90587.66]]

# results - samsung_day2_707122.6875
# loss :  847435.0
# RMSE :  920.5623
# R2 :  0.990690481538419
# samsung_predict :  [[88035.38]]

# results - samsung_day2_670484.4375
# loss :  730228.75
# RMSE :  854.53424
# R2 :  0.9919780540596403
# samsung_predict :  [[88495.86]]

# results - samsung_day2_685991.1875
# loss :  798995.375
# RMSE :  893.86536
# R2 :  0.991222616776648
# samsung_predict :  [[90747.93]]