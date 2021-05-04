import numpy as np
import tensorflow as tf
import autokeras as ak
import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

str_time = datetime.datetime.now()

epoch = 1000
trials = 5

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=23
)

print(x_train.shape) # (455, 30)
print(x_test.shape)  # (114, 30)

model = ak.StructuredDataClassifier(
    max_trials=trials,
    overwrite=True
)

model.fit(
    x_train, y_train,
    epochs=epoch
)

results = model.evaluate(x_test, y_test)

print('trials : {}'.format(trials))
print('epochs : {}'.format(epoch))
print('results : ', results)
print('time : ', datetime.datetime.now() - str_time)

# trials : 2
# epochs : 10
# results :  [0.07261586934328079, 0.9824561476707458]
# time :  0:00:11.715363

# trials : 5
# epochs : 500
# results :  [0.3166152238845825, 0.9736841917037964]
# time :  0:01:07.651087

# trials : 5
# epochs : 1000
# results :  [0.47519055008888245, 0.9736841917037964]
# time :  0:03:10.027603

