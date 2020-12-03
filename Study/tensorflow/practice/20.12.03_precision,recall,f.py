from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

y_true=[1,1,1,0,0,0]
y_pred=[0,1,1,0,0,0]

precision=precision_score(y_true, y_pred)
recall=recall_score(y_true, y_pred)
f1=2*(precision*recall)/(precision+recall)

print('f1:%.3f'%f1)