import numpy as np

from tensorflow.keras.models import load_model

x_train=np.load('../data/npy/samsung_x_train2.npy')
x_test=np.load('../data/npy/samsung_x_test2.npy')
x_val=np.load('../data/npy/samsung_x_val2.npy')
x_pred=np.load('../data/npy/samsung_x_pred2.npy')
y_train=np.load('../data/npy/samsung_y_train2.npy')
y_test=np.load('../data/npy/samsung_y_test2.npy')
y_val=np.load('../data/npy/samsung_y_val2.npy')

model=load_model('../data/modelcheckpoint/samsung_1097667.6250.hdf5')

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_pred)

print('loss : ', loss)
print('samsung_predict: ', y_pred)

# results - samsung_670879.6250
# loss :  966564.0
# samsung_predict:  [[91075.74]]

# results - samsung_649215.8750
# loss :  1008927.625
# samsung_predict:  [[94188.21]]

# results - samsung_662613.9375
# loss :  1147724.0
# samsung_predict:  [[90347.32]]