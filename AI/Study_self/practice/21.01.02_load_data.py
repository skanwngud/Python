# in case of skleran

from sklearn.datasets import load_boston

dataset=load_boston()

x=dataset.data
y=dataset.target

# in case of keras

from tensorflow.keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target)=boston_housing.load_data()

x_train=train_data
x_test=test_data

y_train=train_target
y_test=test_target

