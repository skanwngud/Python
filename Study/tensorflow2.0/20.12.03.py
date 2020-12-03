from sklearn import svm, datasets, model_selection
# 책에서는 cross_validation 이었지만, 0.20 버전부터는 model_selection 을 쓰기로 함
iris=datasets.load_iris()
x=iris.data
y=iris.target

svc=svm.SVC(C=1, kernel='rbf', gamma=0.001)

scores=model_selection.cross_val_score(svc,x,y,cv=5)

print(scores)
print('평균점수:', scores.mean())

#==========================================================================

import numpy as np
from sklearn.metrics import confusion_matrix

y_true=[0,0,0,1,1,1]
y_pred=[1,0,0,1,1,1]

confmat=confusion_matrix(y_true, y_pred)

print(confmat)
#==========================================================================


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('precision:%.3f'%precision_score(y_true, y_pred))
print('recall:%.3f'%recall_score(y_true, y_pred))
print('f1:%.3f'%f1_score(y_true,y_pred))