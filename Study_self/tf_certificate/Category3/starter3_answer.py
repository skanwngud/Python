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
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dense,Activation, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

es=EarlyStopping(
    patience=10,
    verbose=1,
    monitor='val_loss'
)

rl=ReduceLROnPlateau(
    patience=5,
    verbose=1,
    factor=0.5,
    monitor='val_loss'
)

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,
        validation_split=0.2
    )

    validation_datagen=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    # YOUR CODE HERE

    train_generator=training_datagen.flow_from_directory(
        'tmp/rps/',
        subset='training',
        batch_size=32,
        target_size=(150, 150),
    ) # YOUR CODE HERE

    val_generator=validation_datagen.flow_from_directory(
        'tmp/rps/',
        subset='validation',
        batch_size=32,
        target_size=(150, 150)
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, 2, padding='same', input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, 2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(
            learning_rate=0.001
        ),
        metrics='acc'
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=300,
        batch_size=32,
        callbacks=[es, rl]
    )

    loss=model.evaluate(
        val_generator
    )

    print('loss : ', loss[0])
    print('acc : ', loss[1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c:/data/modelcheckpoint/mymodel3.h5")
