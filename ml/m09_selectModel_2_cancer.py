import warnings
warnings.filterwarnings('ignore')

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

from sklearn.datasets import load_breast_cancer

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=33)

all_algorithms=all_estimators(type_filter='classifier')

for (name, algorithm) in all_algorithms:
    try:
        model=algorithm()

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except:
        print(name, '은 없는 모델')

'''

tensorflow acc : 0.9736841917037964

AdaBoostClassifier 의 정답률 :  0.9385964912280702
BaggingClassifier 의 정답률 :  0.9035087719298246
BernoulliNB 의 정답률 :  0.6228070175438597
CalibratedClassifierCV 의 정답률 :  0.9298245614035088
CategoricalNB 의 정답률 :  0.9122807017543859
CheckingClassifier 의 정답률 :  0.37719298245614036
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.8508771929824561
DecisionTreeClassifier 의 정답률 :  0.9035087719298246
DummyClassifier 의 정답률 :  0.4649122807017544
ExtraTreeClassifier 의 정답률 :  0.8508771929824561
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
GaussianNB 의 정답률 :  0.9298245614035088
GaussianProcessClassifier 의 정답률 :  0.9035087719298246
GradientBoostingClassifier 의 정답률 :  0.9298245614035088
HistGradientBoostingClassifier 의 정답률 :  0.9473684210526315
KNeighborsClassifier 의 정답률 :  0.9035087719298246
LabelPropagation 의 정답률 :  0.40350877192982454
LabelSpreading 의 정답률 :  0.40350877192982454
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.9122807017543859
LogisticRegression 의 정답률 :  0.9385964912280702
LogisticRegressionCV 의 정답률 :  0.9736842105263158
MLPClassifier 의 정답률 :  0.9298245614035088
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.8508771929824561
NearestCentroid 의 정답률 :  0.8508771929824561
NuSVC 의 정답률 :  0.8421052631578947
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.8859649122807017
Perceptron 의 정답률 :  0.868421052631579
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9385964912280702
RidgeClassifier 의 정답률 :  0.956140350877193
RidgeClassifierCV 의 정답률 :  0.956140350877193
SGDClassifier 의 정답률 :  0.868421052631579
SVC 의 정답률 :  0.8859649122807017
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''