import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

x_train=x_train/255.
x_test=x_test/255.
x_val=x_val/255.

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

input=Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
cnn1=Conv2D(100, (2,2), padding='same', activation='relu')(input)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(150, (2,2), padding='same', activation='relu')(drop1)
cnn1=Conv2D(200, (2,2), padding='same', activation='relu')(cnn1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(250, (2,2), padding='same', activation='relu')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
flat1=Flatten()(drop1)
dense1=Dense(150, activation='relu')(flat1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(50, activation='relu')(dense1)
output=Dense(10, activation='softmax')(dense1)
model=Model(input, output)

es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp=ModelCheckpoint(filepath='./skanwngud/Study/modelCheckpoint/review_1_{epoch:02d}-{val_loss:.5f}.hdf5',
                    save_best_only=True, monitor='val_loss', mode='auto')
tb=TensorBoard(log_dir='./skanwngud/Study/graph', write_graph=True, write_images=True, histogram_freq=0)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[es, tb, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss[0])
print('acc : ', loss[1])
print('y_test : ', np.argmax(y_test[:5], axis=-1))
print('y_pred : ', np.argmax(y_pred[:5], axis=-1))

# results
# loss :  0.8146396279335022
# acc :  0.7434999942779541
# y_test :  [3 8 8 0 6]
# y_pred :  [3 8 8 0 6]