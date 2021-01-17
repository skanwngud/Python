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

model=load_model('../data/modelcheckpoint/Samsung_day_3_1752028.6250.hdf5')

loss=model.evaluate([x_test, x_1_test], y_test)
y_pred=model.predict([x_test, x_1_test])

pred=model.predict([x_pred, x_1_pred])

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('loss : ', loss)
print('RMSE : ', RMSE(y_test, y_pred))
print('R2 : ', r2_score(y_test, y_pred))
print('Samsung_predict : ', pred)

# results - Samsung_day_3_2007808.2500
# loss :  1938809.625
# RMSE :  1392.4114
# R2 :  0.9802506537095416
# Samsung_predict :  [[95066.42]]

# results - Samsung_day_3_1195530.6250
# loss :  1625298.125
# RMSE :  1274.8724
# R2 :  0.983444165515
# Samsung_predict :  [[94957.74]]

# results - Samsung_day_3_1439483.0000
# loss :  2901311.5
# RMSE :  1703.3236
# R2 :  0.970446297206153
# Samsung_predict :  [[92074.08]]

# results - Samsung_day_3_2233012.0000
# loss :  2122405.75
# RMSE :  1456.8479
# R2 :  0.9783804803298566
# Samsung_predict :  [[94210.59]]

# results - Samsung_day_3_1723690.8750
# loss :  1789911.0
# RMSE :  1337.8756
# R2 :  0.9817673802854208
# Samsung_predict :  [[94671.05]]

# results - Samsung_day_3_1752028.6250
# loss :  1745529.5
# RMSE :  1321.1849
# R2 :  0.9822194674436057
# Samsung_predict :  [[95260.25]]