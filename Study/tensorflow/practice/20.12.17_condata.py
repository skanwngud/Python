import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

train_df=pd.read_csv('/Users/Yoo/Desktop/skanwngud/Study/tensorflow/practice/train.csv')
test_df=pd.read_csv('/Users/Yoo/Desktop/skanwngud/Study/tensorflow/practice/test.csv')
# print(train_df)
train_data=np.array(train_df.iloc[:,1:], dtype='float32')
test_data=np.array(test_df.iloc[:,1:], dtype='float32')
# print(train_data)
x_train=train_data[:,1:]/255
# x_train=train_data[:1.:]/255 에서 수정함
y_train=train_data[:,0]
x_test=test_data/255

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=12345)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test=x_test.reshape(x_test.shape[0], 28, 28, 1)
x_validate=x_validate.reshape(x_validate.shape[0], 28, 28, 1)

cnn_model=Sequential([Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=2), Flatten(), Dense(32, activation = 'relu'), Dense(10, activation= 'softmax')])

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

hystory = cnn_model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=20,
    verbose=1,
    validation_data=(x_validate, y_validate),
)

y_pred = cnn_model.predict_classes(x_test)

submission = pd.read_csv('/Users/Yoo/Desktop/skanwngud/Study/tensorflow/practice/sample_submission.csv', encoding='uft-8')
submission['label']=y_pred
submission.to_csv('fashion_submission.csv', index= False)