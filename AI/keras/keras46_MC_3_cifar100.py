import numpy as np

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test)=cifar100.load_data()
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=56)

print(x_train.shape) # (500000, 32, 32, 3)
print(y_train.shape) # (500000, 1)

x_train=x_train/255.
x_test=x_test/255.
x_val=x_val/255.

enc=OneHotEncoder()
enc.fit(y_train)
y_train=enc.transform(y_train).toarray()
y_test=enc.transform(y_test).toarray()
y_val=enc.transform(y_val).toarray()

input1=Input(shape=(32, 32, 3))
cnn1=Conv2D(100, (2,2), padding='same')(input1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(150, (2,2), padding='same')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
cnn1=Conv2D(200, (2,2), padding='same')(drop1)
max1=MaxPooling2D((2,2))(cnn1)
drop1=Dropout(0.2)(max1)
flat1=Flatten()(drop1)
dense1=Dense(100, activation='relu')(flat1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(300, activation='relu')(dense1)
dense1=Dense(250, activation='relu')(dense1)
dense1=Dense(200, activation='relu')(dense1)
dense1=Dense(150, activation='relu')(dense1)
dense1=Dense(100, activation='relu')(dense1)
dense1=Dense(80, activation='relu')(dense1)
dense1=Dense(60, activation='relu')(dense1)
dense1=Dense(30, activation='relu')(dense1)
output1=Dense(100, activation='softmax')(dense1)
model=Model(input1, output1)

cp=ModelCheckpoint(filepath='../data/modelcheckpoint/k46_3_cifar100_{epoch:02d}-{val_loss:.4f}.hdf5',
                    monitor='val_loss', save_best_only=True, mode='auto')
early=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), callbacks=[early, cp])

loss=model.evaluate(x_test, y_test)
y_pred=model.predict(x_test)

print('loss : ', loss)
print('y_pred : ', np.argmax(y_pred[:5], axis=-1))
print('y_test : ', np.argmax(y_test[:5], axis=-1))

# results
# loss :  [1.2142846584320068, 0.714900016784668]
# y_pred :  [3 1 8 0 6]
# y_test :  [3 8 8 0 6]

# results
# loss :  [1.3180079460144043, 0.7196999788284302]
# y_pred :  [5 1 8 0 6]
# y_test :  [3 8 8 0 6]

# results
# loss :  [1.3434417247772217, 0.7164000272750854]
# y_pred :  [3 8 8 2 6]
# y_test :  [3 8 8 0 6]