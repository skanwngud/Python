import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score, mean_squared_error

# datasets=np.load('../data/npy/samsung_data_2.npz')
datasets=np.load('../data/npy/samsung_data_2.npz')

x_train=datasets['x_train']
x_test=datasets['x_test']
x_val=datasets['x_val']
x_pred=datasets['x_pred']

y_train=datasets['y_train']
y_test=datasets['y_test']
y_val=datasets['y_val']

model=load_model('../data/modelcheckpoint/samsung_day_2_718449.8125.hdf5')

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

# results - samsung_805872.7500
# loss :  835469.125
# RMSE :  914.0403
# R2 :  0.9911528186096876
# samsung_predict :  [[90224.63]]

# results - samsung_day_2_777621.8125
# loss :  823142.0
# RMSE :  907.2716
# R2 :  0.9912833639205951
# samsung_predict :  [[89692.04]]

# results - samsung_day_2_733243.6250
# loss :  848947.75
# RMSE :  921.3842
# R2 :  0.9910100815271781
# samsung_predict :  [[89197.77]]

# results - samsung_day_2_718449.8125
# loss :  895634.1875
# RMSE :  946.37933
# R2 :  0.9905157123523909
# samsung_predict :  [[89203.83]]