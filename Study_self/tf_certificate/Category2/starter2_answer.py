# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
import tensorflow
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dense, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

es=EarlyStopping(
    patience=5,
    verbose=1
)

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

    x_train, x_val, y_train, y_val=train_test_split(
        x_train, y_train,
        train_size=0.8,
        random_state=23
    )

    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    y_val=to_categorical(y_val)

    x_train=x_train.reshape(-1, 28, 28, 1)/255.
    x_test=x_test.reshape(-1, 28, 28, 1)/255.
    x_val=x_val.reshape(-1, 28, 28, 1)/255.

    model=Sequential()
    model.add(Conv2D(128, 2, padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics='acc'
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es]
    )

    loss=model.evaluate(
        x_test, y_test
    )

    pred=model.predict(
        x_test
    )

    print('pred : ', np.argmax(pred[:5], axis=-1))
    print('x_test : ', np.argmax(y_test[:5], axis=-1))

    # YOUR CODE HERE
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c:/data/modelcheckpoint/mymodel2.h5")
