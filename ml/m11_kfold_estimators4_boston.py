import warnings
warnings.filterwarnings('ignore') # 경고문 무시

import sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정

from sklearn.datasets import load_boston

datasets=load_boston()
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
 [0.75559186 0.71548987 0.73898792 0.47602642 0.6529103 ]
AdaBoostRegressor 의 정답률 : 
 [0.87106246 0.88173846 0.84162251 0.81933612 0.7482997 ]
BaggingRegressor 의 정답률 : 
 [0.89678022 0.89305267 0.83754283 0.73007819 0.83612058]
BayesianRidge 의 정답률 :
 [0.74653397 0.73254151 0.73421701 0.53828908 0.64593235]
CCA 의 정답률 : 
 [0.64661117 0.70985344 0.72481331 0.18994707 0.67563018]
DecisionTreeRegressor 의 정답률 : 
 [0.8007322  0.86906723 0.60485261 0.04741957 0.71852592]
DummyRegressor 의 정답률 :
 [-0.01233695 -0.00272558 -0.0001637  -0.07764968 -0.09183118]
ElasticNet 의 정답률 :
 [0.69976525 0.68352794 0.67460256 0.62751633 0.58117346]
ElasticNetCV 의 정답률 : 
 [0.69085912 0.66895175 0.6659745  0.61975846 0.56371034]
ExtraTreeRegressor 의 정답률 :
 [0.78882897 0.76721275 0.85081405 0.16108251 0.6170506 ]
ExtraTreesRegressor 의 정답률 : 
 [0.92110959 0.90310241 0.91212128 0.80196484 0.77872527]
GammaRegressor 의 정답률 :
 [-0.01097991 -0.00275879 -0.00015737 -0.07780167 -0.11084548]
GaussianProcessRegressor 의 정답률 : 
 [-6.72156689 -5.36622287 -6.58081538 -8.71808818 -4.72159521]
GeneralizedLinearRegressor 의 정답률 : 
 [0.6654751  0.666324   0.65424658 0.55959755 0.59880889]
GradientBoostingRegressor 의 정답률 : 
 [0.88373996 0.91784071 0.90410835 0.78510135 0.77772588]
HistGradientBoostingRegressor 의 정답률 : 
 [0.90148993 0.86427371 0.87232714 0.79586539 0.79177114]
HuberRegressor 의 정답률 : 
 [0.60388978 0.5427462  0.70095165 0.41358948 0.5277654 ]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :
 [0.54394853 0.45030555 0.58895745 0.28657694 0.30497472]
KernelRidge 의 정답률 : 
 [0.74096636 0.73485133 0.7469715  0.42266291 0.6595487 ]
Lars 의 정답률 :
 [0.75200677 0.74039615 0.7439188  0.53368041 0.65960426]
LarsCV 의 정답률 : 
 [0.75310258 0.74039615 0.75553916 0.52208257 0.65087477]
Lasso 의 정답률 :
 [0.70356026 0.67976574 0.67397349 0.61322953 0.56612265]
LassoCV 의 정답률 : 
 [0.72813118 0.70031834 0.70411048 0.59957556 0.5997275 ]
LassoLars 의 정답률 :
 [-0.01233695 -0.00272558 -0.0001637  -0.07764968 -0.09183118]
LassoLarsCV 의 정답률 : 
 [0.75663017 0.74039615 0.75553916 0.52208257 0.65495319]
LassoLarsIC 의 정답률 :
 [0.75507628 0.73848679 0.75761155 0.45962562 0.65619561]
LinearRegression 의 정답률 : 
 [0.75374442 0.74039615 0.7439188  0.53368041 0.65960426]
LinearSVR 의 정답률 : 
 [0.50508486 0.45704906 0.66077176 0.41283392 0.63367544]
MLPRegressor 의 정답률 : 
 [0.51374588 0.43281099 0.54390102 0.57477911 0.47322304]
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
 [ 0.3349288   0.15841151  0.33297666  0.2832023  -0.01927449]
OrthogonalMatchingPursuit 의 정답률 :
 [ 0.57932393  0.51599931  0.51590512 -0.10671003  0.49617566]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.75875635 0.68940315 0.7259702  0.4027586  0.6045469 ]
PLSCanonical 의 정답률 :
 [-2.98841245 -0.8519351  -2.07588368 -5.86782105 -0.92435686]
PLSRegression 의 정답률 : 
 [0.77178738 0.70600206 0.78664355 0.49782479 0.61073635]
PassiveAggressiveRegressor 의 정답률 :
 [-3.75310363 -0.15747403  0.16050364 -0.33578249 -0.33876761]
PoissonRegressor 의 정답률 : 
 [0.83013313 0.7874634  0.76308179 0.62146444 0.74510119]
RANSACRegressor 의 정답률 : 
 [0.69170821 0.62758654 0.72296019 0.03452701 0.38378583]
RadiusNeighborsRegressor 은 없는 모델
RandomForestRegressor 의 정답률 : 
 [0.91593163 0.91992859 0.8792053  0.6733538  0.78167887]
RegressorChain 은 없는 모델
Ridge 의 정답률 :
 [0.75313747 0.73901878 0.74770609 0.52160775 0.65579243]
RidgeCV 의 정답률 :
 [0.75398069 0.74054465 0.74508776 0.53135909 0.65916663]
SGDRegressor 의 정답률 : 
 [-1.09682267e+27 -8.59352316e+25 -6.47150009e+26 -1.52565162e+27
 -1.01391324e+25]
SVR 의 정답률 : 
 [ 0.30513599  0.14080387  0.28070801  0.2772936  -0.0510085 ]
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 : 
 [0.76206281 0.72990892 0.78296062 0.45703744 0.60113492]
TransformedTargetRegressor 의 정답률 :
 [0.75374442 0.74039615 0.7439188  0.53368041 0.65960426]
TweedieRegressor 의 정답률 : 
 [0.6654751  0.666324   0.65424658 0.55959755 0.59880889]
VotingRegressor 은 없는 모델
_SigmoidCalibration 의 정답률 :
 [nan nan nan nan nan]
'''