import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_diabetes

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.8, random_state=33)

kf=KFold(n_splits=5, shuffle=True, random_state=24)

allAlgorithms=all_estimators(type_filter='regressor') # 분류형 모델 전체
# all_estimators - sklearn 0.20.0 대에 최적화

for (name, algorithm) in allAlgorithms: # name : 분류모델의 이름, algorithm : 분류모델
    try:
        model=algorithm()

        score=cross_val_score(model, x_train, y_train, cv=kf) # 위에서 n_splits=5 로 했기 때문에 cv=5 라고 해도 된다
        print(name, '의 정답률 : \n', score)
    except: # 예외 발생시 실행 할 행동
        print(name, '은 없는 모델') # continue 를 쓰면 무시하고 루프를 계속 돌게 됨

'''
ARDRegression 의 정답률 : 
 [0.37969962 0.46558325 0.45918263 0.45744323 0.53450956]
AdaBoostRegressor 의 정답률 : 
 [0.30832996 0.3044151  0.36996438 0.36161724 0.42905064]
BaggingRegressor 의 정답률 : 
 [0.37988473 0.30387374 0.42712675 0.30617937 0.3923646 ]
BayesianRidge 의 정답률 :
 [0.36100417 0.46436869 0.46175734 0.44662394 0.54579434]
CCA 의 정답률 :
 [0.39506889 0.45471672 0.4547334  0.47621631 0.4841932 ]
DecisionTreeRegressor 의 정답률 : 
 [ 0.04291905 -0.06316237 -0.17849967 -0.17529354 -0.07448103]
DummyRegressor 의 정답률 :
 [-0.05588753 -0.08481631 -0.01138309 -0.00409008 -0.00425781]
ElasticNet 의 정답률 :
 [-0.04693852 -0.07703638 -0.0032232   0.00441207  0.00391184]
ElasticNetCV 의 정답률 : 
 [0.36335153 0.39220757 0.4080793  0.40500491 0.47177601]
ExtraTreeRegressor 의 정답률 :
 [-0.58889688 -0.17662382 -0.49899636 -0.20640095 -0.04484789]
ExtraTreesRegressor 의 정답률 : 
 [0.38186041 0.36888352 0.47005331 0.39526398 0.428898  ]
GammaRegressor 의 정답률 :
 [-0.03913247 -0.07426618 -0.00460378  0.00184104  0.00191511]
GaussianProcessRegressor 의 정답률 : 
 [-10.88515829 -15.78888065  -6.18243743 -39.59140939 -18.87549604]
GeneralizedLinearRegressor 의 정답률 :
 [-0.04908133 -0.07895962 -0.00541352  0.00202076  0.00204022]
GradientBoostingRegressor 의 정답률 : 
 [0.38334072 0.31531015 0.37352328 0.34680208 0.4073999 ]
HistGradientBoostingRegressor 의 정답률 : 
 [0.36018126 0.25813014 0.37814872 0.37712447 0.38460602]
HuberRegressor 의 정답률 : 
 [0.3677837  0.45826607 0.45182669 0.42282989 0.53717164]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :
 [0.36768456 0.32479773 0.40659324 0.2497292  0.46547697]
KernelRidge 의 정답률 :
 [-3.6326983  -3.87376307 -4.08655725 -3.69456747 -3.64661073]
Lars 의 정답률 : 
 [ 0.36682794 -2.11448604  0.45811215  0.40081172  0.53735957]
LarsCV 의 정답률 : 
 [0.36682794 0.45291243 0.45836759 0.44809227 0.53909975]
Lasso 의 정답률 :
 [0.29941783 0.26007638 0.30662686 0.35929135 0.30581307]
LassoCV 의 정답률 : 
 [0.36661782 0.47645805 0.46422628 0.44768217 0.53798087]
LassoLars 의 정답률 :
 [0.34272922 0.31428524 0.34876835 0.39667459 0.3416607 ]
LassoLarsCV 의 정답률 : 
 [0.36682794 0.47772211 0.46388436 0.44830876 0.53735957]
LassoLarsIC 의 정답률 :
 [0.36429274 0.46623885 0.46479258 0.44619411 0.53638089]
LinearRegression 의 정답률 :
 [0.36682794 0.47772211 0.46388436 0.46835968 0.53735957]
LinearSVR 의 정답률 :
 [-0.24613085 -0.78389499 -0.37909478 -0.46701824 -0.53839169]
MLPRegressor 의 정답률 : 
 [-2.81576398 -3.48253829 -3.14094983 -3.30873744 -3.15191886]
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 의 정답률 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 : 
 [nan nan nan nan nan]
NuSVR 의 정답률 : 
 [0.11332522 0.00938455 0.13458876 0.12838159 0.12254573]
OrthogonalMatchingPursuit 의 정답률 :
 [0.24367829 0.36303933 0.23116432 0.39494935 0.28733973]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.37827694 0.44119513 0.42320081 0.42124901 0.51358024]
PLSCanonical 의 정답률 :
 [-2.0979638  -0.34079171 -1.31146487 -2.32761754 -0.8081124 ]
PLSRegression 의 정답률 :
 [0.36611441 0.46055101 0.4634033  0.4307735  0.55958186]
PassiveAggressiveRegressor 의 정답률 : 
 [0.39121943 0.42753453 0.38499564 0.39528614 0.5123499 ]
PoissonRegressor 의 정답률 : 
 [0.27149589 0.26247321 0.30273945 0.30119115 0.35744939]
RANSACRegressor 의 정답률 : 
 [0.02191822 0.32815513 0.16985648 0.34119494 0.12855483]
RadiusNeighborsRegressor 의 정답률 :
 [-0.05588753 -0.08481631 -0.01138309 -0.00409008 -0.00425781]
RandomForestRegressor 의 정답률 : 
 [0.3662218  0.33881518 0.36892854 0.43923917 0.42433642]
RegressorChain 은 없는 모델
Ridge 의 정답률 :
 [0.3392087  0.33451848 0.37088776 0.36876091 0.42493621]
RidgeCV 의 정답률 :
 [0.37068256 0.45658142 0.46042305 0.44896885 0.54279002]
SGDRegressor 의 정답률 : 
 [0.32666215 0.32627356 0.3517686  0.33470436 0.41348904]
SVR 의 정답률 : 
 [ 0.16366823 -0.03904946  0.14202808  0.15253665  0.11429596]
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 : 
 [0.3694662  0.45593932 0.4642727  0.43979742 0.52672317]
TransformedTargetRegressor 의 정답률 :
 [0.36682794 0.47772211 0.46388436 0.46835968 0.53735957]
TweedieRegressor 의 정답률 :
 [-0.04908133 -0.07895962 -0.00541352  0.00202076  0.00204022]
VotingRegressor 은 없는 모델
_SigmoidCalibration 의 정답률 : 
 [nan nan nan nan nan]
'''