import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

print(x.shape) # (569, 30)

for i in range(1, 31):
    pca=PCA(n_components=i)
    x2=pca.fit_transform(x)

    model=RandomForestClassifier()
    model.fit(x2, y)
    
    acc=model.score(x2, y)
    print(str(i) + '번째 varience_ratio', np.cumsum(pca.explained_variance_ratio_))
    print(str(i) + '번째 acc : ', acc)

# results
# 1번째 acc :  1.0
# 2번째 acc :  1.0
# 3번째 acc :  1.0
# 4번째 acc :  1.0
# 5번째 acc :  1.0
# 6번째 acc :  1.0
# 7번째 acc :  1.0
# 8번째 acc :  1.0
# 9번째 acc :  1.0
# 10번째 acc :  1.0
# 11번째 acc :  1.0
# 12번째 acc :  1.0
# 13번째 acc :  1.0
# 14번째 acc :  1.0
# 15번째 acc :  1.0
# 16번째 acc :  1.0
# 17번째 acc :  1.0
# 18번째 acc :  1.0
# 19번째 acc :  1.0
# 20번째 acc :  1.0
# 21번째 acc :  1.0
# 22번째 acc :  1.0
# 23번째 acc :  1.0
# 24번째 acc :  1.0
# 25번째 acc :  1.0
# 26번째 acc :  1.0
# 27번째 acc :  1.0
# 28번째 acc :  1.0
# 29번째 acc :  1.0
# 30번째 acc :  1.0