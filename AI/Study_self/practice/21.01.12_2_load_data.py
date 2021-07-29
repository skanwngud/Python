import numpy as np

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping

x_train=np.load('../data/npy/review2_x_train.npy')
x_test=np.load('../data/npy/review2_x_test.npy')
x_val=np.load('../data/npy/review2_x_val.npy')
y_train=np.load('../data/npy/review2_y_train.npy')
y_test=np.load('../data/npy/review2_y_test.npy')
y_val=np.load('../data/npy/review2_y_val.npy')

# input=Input(shape=(x_train.shape[1], x_train.shape[2], 1))
# cnn1=Conv2D(10, (2,2), padding='same', activation='relu')(input)
# max1=MaxPooling2D((2,2))(cnn1)
# drop1=Dropout(0.2)(max1)
# cnn1=Conv2D(9, (2,2), padding='same', activation='relu')(drop1)
# max1=MaxPooling2D((2,2))(cnn1)
# drop1=Dropout(0.2)(max1)
# flat1=Flatten()(drop1)
# dense1=Dense(150, activation='relu')(flat1)
# dense1=Dense(200, activation='relu')(dense1)
# dense1=Dense(250, activation='relu')(dense1)
# dense1=Dense(200, activation='relu')(dense1)
# dense1=Dense(150, activation='relu')(dense1)
# dense1=Dense(100, activation='relu')(dense1)
# dense1=Dense(80, activation='relu')(dense1)
# dense1=Dense(60, activation='relu')(dense1)
# output=Dense(10, activation='softmax')(dense1)
# model=Model(input, output)

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

print(x_train.shape)
print(y_train.shape)
# model1=load_model('../data/h5/review2_1_save_model.h5')

# es=EarlyStopping(monitor='val_loss', mode='auto', patience=10)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val, y_val), callbacks=es)

# model=load_model('../data/modelcheckpoint/review2_1_modelcheckpoint_30-0.0583.hdf5')
# loss=model.evaluate(x_test, y_test)

# print('checkpoint_loss : ', loss[0])
# print('checkpoint_acc : ', loss[1])


# loss1=model.evaluate(x_test, y_test)

# print('save_model1_loss : ', loss1[0])
# print('save_model1_acc : ', loss1[1])

# model2=load_model('../data/h5/review2_2_save_model.h5')
# loss2=model2.evaluate(x_test, y_test)

# print('save_model2_loss : ', loss2[0])
# print('save_model2_acc : ', loss2[1])

model.load_weights('../data/h5/review2_3_save_weight.h5')
loss3=model.evaluate(x_test, y_test)

print('weight_loss : ', loss3[0])
print('weight_acc : ', loss3[1])

# results
# checkpoint_loss :  0.061146125197410583
# checkpoint_acc :  0.9818000197410583
# save_model1_loss :  0.05724416673183441
# save_model1_acc :  0.9814000129699707
# save_model2_loss :  0.061146125197410583
# save_model2_acc :  0.9818000197410583
# weight_loss :  0.061146125197410583
# weight_acc :  0.9818000197410583