import numpy as np

from tensorflow.keras.models import load_model

x_train=np.load('../data/npy/samsung_x_train.npy')
x_test=np.load('../data/npy/samsung_x_test.npy')
x_val=np.load('../data/npy/samsung_x_val.npy')
x_pred=np.load('../data/npy/samsung_x_pred.npy')
y_train=np.load('../data/npy/samsung_y_train.npy')
y_test=np.load('../data/npy/samsung_y_test.npy')
y_val=np.load('../data/npy/samsung_y_val.npy')

model=load_model('../data/modelcheckpoint/samsung_93-10092972.0000.hdf5')

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_pred)

print('loss : ', loss)
print('samsung_predict: ', y_pred)