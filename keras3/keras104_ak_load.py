import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = load_model(
    'c:/data/h5/autokeras_1.h5'
)

model.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
cast_to_float32 (CastToFloat (None, 28, 28, 1)         0
_________________________________________________________________
normalization (Normalization (None, 28, 28, 1)         3
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 9216)              0
_________________________________________________________________
dense (Dense)                (None, 10)                92170
_________________________________________________________________
classification_head_1 (Softm (None, 10)                0
=================================================================
Total params: 110,989
Trainable params: 110,986
Non-trainable params: 3
_________________________________________________________________

mnist data 에서 autokeras 가 가장 좋았다고 판단한 파라미터
'''

# model = ak.ImageClassifier( 
#     overwrite=True,       
#     max_trials=1,
#     loss='mse',
#     metrics=['acc']
# )

# model.summary()


# es = EarlyStopping(
#     verbose=1, patience=10
# )
# rl = ReduceLROnPlateau(
#     verbose=1, patience=5
# )
# mc = ModelCheckpoint(
#     './temp/',
#     verbose=1, save_best_only=True
# )

# model.fit(
#     x_train, y_train,
#     epochs=1,
#     validation_split=0.2, 
#     callbacks=[es, rl, mc]
# )
# # default - validation_split = 0.2
# # callbacks can use

# results = model.evaluate(x_test, y_test)

# print(results)

# # model.summary()

# model2=model.export_model()
# model2.save('c:/data/h5/autokeras_1.h5')

