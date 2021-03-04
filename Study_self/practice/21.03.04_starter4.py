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
    patience=30,
    verbose=1
)

rl=tf.keras.callbacks.ReduceLROnPlateau(
    patience=10,
    verbose=1,
    factor=0.5
)

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

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
with open('sarcasm.json', 'r') as f:
    datasets=json.load(f)
for item in datasets:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

    x_train=sentences[0:training_size]
    x_test=sentences[training_size:]
    y_train=labels[0:training_size]
    y_test=labels[training_size:]

token=Tokenizer()
x_train=token.texts_to_sequences(x_train)
x_test=token.texts_to_sequences(x_test)


x_train=pad_sequences(
    x_train,
    maxlen=max_length
)

x_test=pad_sequences(
    x_test,
    maxlen=max_length
)

print(sentences)
print(x_train)
print(x_test)

print(y_train)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length),
    tf.keras.layers.Conv1D(128, 2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.LSTM(128, activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
# YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics='acc'
)

model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    callbacks=[es, rl]
)


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