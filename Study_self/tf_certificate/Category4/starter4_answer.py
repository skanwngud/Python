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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

es=tf.keras.callbacks.EarlyStopping(
    patience=20,
    verbose=1
)

rl=tf.keras.callbacks.ReduceLROnPlateau(
    patience=10,
    verbose=1,
    factor=0.5
)

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'C:\ML\sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('C:\ML\sarcasm.json', 'r') as f:
        datasets=json.load(f)
    for item in datasets:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    x_train=sentences[0:training_size]
    x_test=sentences[training_size:]
    y_train=labels[0:training_size]
    y_test=labels[training_size:]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    token=Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    token.fit_on_texts(x_train)
    token.fit_on_texts(x_test)

    x_train=token.texts_to_sequences(x_train)
    x_test=token.texts_to_sequences(x_test)

    pad_x_train=pad_sequences(
        x_train,
        padding=padding_type,
        truncating=trunc_type,
        maxlen=max_length
    )

    pad_x_test=pad_sequences(
        x_test,
        padding=padding_type,
        truncating=trunc_type,
        maxlen=max_length
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=training_size,
            output_dim=embedding_dim,
            input_length=max_length),
        tf.keras.layers.Conv1D(64, 2),
        # tf.keras.layers.Conv1D(64, 2),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
<<<<<<< HEAD
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
=======
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(32, activation='relu'),
>>>>>>> 3b84ef7be9b8026d58ffde4b6221d535f63d6db0
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics='acc'
    )

    model.fit(
        pad_x_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[es, rl]
    )

    loss=model.evaluate(
        pad_x_test, y_test
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
    model.save("c:/data/modelcheckpoint/mymodel4.h5")


'''
with open('sarcasm.json', 'r') as f:
    datasets=json.load(f)
for item in datasets:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

x_train=sentences[0:training_size]
x_test=sentences[training_size:]
y_train=labels[0:training_size]
y_test=labels[training_size:]
'''