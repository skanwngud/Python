import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_breast_cancer

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=33)

kf=KFold(n_splits=5, shuffle=True, random_state=24)

allAlgorithms=all_estimators(type_filter='classifier') # 분류형 모델 전체
# all_estimators - sklearn 0.20.0 대에 최적화

for (name, algorithm) in allAlgorithms: # name : 분류모델의 이름, algorithm : 분류모델
    try:
        model=algorithm()

        score=cross_val_score(model, x_train, y_train, cv=kf) # 위에서 n_splits=5 로 했기 때문에 cv=5 라고 해도 된다
        print(name, '의 정답률 : \n', score)
    except: # 예외 발생시 실행 할 행동
        print(name, '은 없는 모델') # continue 를 쓰면 무시하고 루프를 계속 돌게 됨

''''
AdaBoostClassifier 의 정답률 : 
 [0.93406593 0.98901099 0.96703297 0.95604396 0.96703297]
BaggingClassifier 의 정답률 : 
 [0.91208791 0.93406593 0.95604396 0.92307692 0.95604396]
BernoulliNB 의 정답률 :
 [0.6043956  0.65934066 0.67032967 0.67032967 0.53846154]
CalibratedClassifierCV 의 정답률 : 
 [0.93406593 0.92307692 0.91208791 0.94505495 0.93406593]
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :
 [0.89010989 0.92307692 0.93406593 0.87912088 0.92307692]
DecisionTreeClassifier 의 정답률 : 
 [0.93406593 0.94505495 0.9010989  0.93406593 0.96703297]
DummyClassifier 의 정답률 :
 [0.57142857 0.47252747 0.52747253 0.52747253 0.47252747]
ExtraTreeClassifier 의 정답률 :
 [0.89010989 0.96703297 0.9010989  0.94505495 0.93406593]
ExtraTreesClassifier 의 정답률 : 
 [0.95604396 0.98901099 0.96703297 0.96703297 0.96703297]
GaussianNB 의 정답률 :
 [0.92307692 0.98901099 0.92307692 0.93406593 0.94505495]
GaussianProcessClassifier 의 정답률 : 
 [0.9010989  0.95604396 0.94505495 0.94505495 0.85714286]
GradientBoostingClassifier 의 정답률 : 
 [0.96703297 0.97802198 0.95604396 0.93406593 0.98901099]
HistGradientBoostingClassifier 의 정답률 : 
 [0.94505495 0.98901099 0.98901099 0.95604396 0.97802198]
KNeighborsClassifier 의 정답률 : 
 [0.93406593 0.97802198 0.93406593 0.95604396 0.91208791]
LabelPropagation 의 정답률 : 
 [0.41758242 0.35164835 0.32967033 0.35164835 0.47252747]
LabelSpreading 의 정답률 : 
 [0.41758242 0.35164835 0.32967033 0.35164835 0.47252747]
LinearDiscriminantAnalysis 의 정답률 : 
 [0.93406593 0.97802198 0.96703297 0.94505495 0.93406593]
LinearSVC 의 정답률 : 
 [0.92307692 0.92307692 0.91208791 0.92307692 0.93406593]
LogisticRegression 의 정답률 : 
 [0.91208791 0.94505495 0.96703297 0.97802198 0.94505495]
LogisticRegressionCV 의 정답률 : 
 [0.91208791 0.98901099 0.96703297 0.98901099 0.95604396]
MLPClassifier 의 정답률 : 
 [0.91208791 0.94505495 0.91208791 0.95604396 0.93406593]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :
 [0.89010989 0.92307692 0.93406593 0.87912088 0.92307692]
NearestCentroid 의 정답률 :
 [0.89010989 0.95604396 0.86813187 0.87912088 0.89010989]
NuSVC 의 정답률 : 
 [0.85714286 0.94505495 0.86813187 0.87912088 0.87912088]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :
 [0.91208791 0.9010989  0.94505495 0.92307692 0.9010989 ]
Perceptron 의 정답률 : 
 [0.93406593 0.87912088 0.87912088 0.82417582 0.9010989 ]
QuadraticDiscriminantAnalysis 의 정답률 :
 [0.94505495 0.97802198 0.96703297 0.94505495 0.95604396]
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 : 
 [0.96703297 0.98901099 0.96703297 0.95604396 0.95604396]
RidgeClassifier 의 정답률 :
 [0.94505495 0.97802198 0.95604396 0.96703297 0.93406593]
RidgeClassifierCV 의 정답률 : 
 [0.95604396 0.96703297 0.95604396 0.95604396 0.92307692]
SGDClassifier 의 정답률 :
 [0.9010989  0.9010989  0.9010989  0.69230769 0.92307692]
SVC 의 정답률 : 
 [0.89010989 0.95604396 0.93406593 0.92307692 0.9010989 ]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''