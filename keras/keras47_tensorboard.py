random_seed=66

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

modelpath='../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early=EarlyStopping(monitor='val_loss', patience=5, mode='auto')
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb=TensorBoard(log_dir='../data/graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist=model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early, tb])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print(loss)
print(np.argmax(y_pred[:10], axis=-1))
print(np.argmax(y_test[:10], axis=-1))

plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

'''
코드 실행 후 커맨드 창에서

cd \
cd skanwngud
cd Study
cd graph
dir/w
tensorboard --logdir=.

http://127.0.0.1:6006
'''

# results
# [0.0630539208650589, 0.9883000254631042]
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]