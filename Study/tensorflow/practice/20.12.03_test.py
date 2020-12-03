import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)

svc=svm.SVC(C=1, kernel='rbf', gamma=0.001)
svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)
print('accuracy:%.2f'%accuracy_score(y_test, y_pred))