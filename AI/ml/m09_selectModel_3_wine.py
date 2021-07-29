import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

from sklearn.datasets import load_wine

datasets=load_wine()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.8, random_state=22)

all_algorithms=all_estimators(type_filter='classifier')

for (name, algorithm) in all_algorithms:
    try:
        model=algorithm()

        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        print(name, '의 정답률', accuracy_score(y_test, y_pred))

    except:
        print(name, '은 없는 모델')

'''

tensorflow acc : 1.0

AdaBoostClassifier 의 정답률 0.9722222222222222
BaggingClassifier 의 정답률 0.9166666666666666
BernoulliNB 의 정답률 0.4444444444444444
CalibratedClassifierCV 의 정답률 0.8888888888888888
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 0.25
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 0.6388888888888888
DecisionTreeClassifier 의 정답률 0.8888888888888888
DummyClassifier 의 정답률 0.3333333333333333
ExtraTreeClassifier 의 정답률 0.8055555555555556
ExtraTreesClassifier 의 정답률 0.9444444444444444
GaussianNB 의 정답률 0.9722222222222222
GaussianProcessClassifier 의 정답률 0.3888888888888889
GradientBoostingClassifier 의 정답률 0.9166666666666666
HistGradientBoostingClassifier 의 정답률 0.9722222222222222
KNeighborsClassifier 의 정답률 0.6944444444444444
LabelPropagation 의 정답률 0.3333333333333333
LabelSpreading 의 정답률 0.3333333333333333
LinearDiscriminantAnalysis 의 정답률 1.0
LinearSVC 의 정답률 0.8611111111111112
LogisticRegression 의 정답률 0.9166666666666666
LogisticRegressionCV 의 정답률 0.9166666666666666
MLPClassifier 의 정답률 0.9166666666666666
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 0.8333333333333334
NearestCentroid 의 정답률 0.7777777777777778
NuSVC 의 정답률 0.8611111111111112
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 0.6388888888888888
Perceptron 의 정답률 0.6111111111111112
QuadraticDiscriminantAnalysis 의 정답률 1.0
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 0.9722222222222222
RidgeClassifier 의 정답률 1.0
RidgeClassifierCV 의 정답률 1.0
SGDClassifier 의 정답률 0.5
SVC 의 정답률 0.6388888888888888
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''