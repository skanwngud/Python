import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=12)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

np.save('../data/npy/review2_x_train.npy', arr=x_train)
np.save('../data/npy/review2_x_test.npy', arr=x_test)
np.save('../data/npy/review2_x_val.npy', arr=x_val)
np.save('../data/npy/review2_y_train.npy', arr=y_train)
np.save('../data/npy/review2_y_test.npy', arr=y_test)
np.save('../data/npy/review2_y_val.npy', arr=y_val)

input=Input(shape=(x_train.shape[1], x_train.shape[2], 1))
cnn1=Conv2D(10, (2,2), padding='same')(input)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(9, (2,2), padding='same')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(8, (2,2), padding='same')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
flat1=Flatten()(drop1)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(120, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
output=Dense(10, activation='softmax')(dense1)
model=Model(input, output)

model.save('../data/h5/review2_1_save_model.h5')

es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp=ModelCheckpoint(filepath='../data/modelcheckpoint/review2_1_modelcheckpoint_{epoch:02d}-{val_loss:.4f}.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val, y_val), callbacks=[es, cp])

model.save('../data/h5/review2_2_save_model.h5')
model.save_weights('../data/h5/review2_3_save_weight.h5')

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

# results
# loss :  0.05806514248251915
# acc :  0.9825999736785889