import numpy as np

from tensorflow.keras.models import load_model

x_train=np.load('../data/npy/samsung_x_train.npy')
x_test=np.load('../data/npy/samsung_x_test.npy')
x_val=np.load('../data/npy/samsung_x_val.npy')
x_pred=np.load('../data/npy/samsung_x_pred.npy')
y_train=np.load('../data/npy/samsung_y_train.npy')
y_test=np.load('../data/npy/samsung_y_test.npy')
y_val=np.load('../data/npy/samsung_y_val.npy')

model=load_model('../data/modelcheckpoint/samsung_600948.7500.hdf5')

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_pred)

print('loss : ', loss)
print('samsung_predict: ', y_pred)

# results
# loss :  1122932.875
# samsung_predict:  [[[92979.8]]]

# results - samsung_601974.1875
# loss :  1277852.125
# samsung_predict:  [[89809.67]]

# results - samsung_600948.7500
# loss :  1232579.625
# samsung_predict:  [[90150.57]]