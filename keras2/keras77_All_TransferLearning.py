import tensorflow
import time

from keras.applications import VGG16, VGG19, Xception
from keras.applications import ResNet101, ResNet101V2, \
    ResNet152, ResNet152V2, ResNet50, ResNet50V2
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
# from keras.applications import EfficientNetB0~7 (설치 필요)

Model=[VGG16(), VGG19(), Xception(), ResNet101(), ResNet101V2(), ResNet152(),
ResNet152V2(), ResNet50(), ResNet50V2(), InceptionV3(), InceptionResNetV2(), MobileNet(),
MobileNetV2(), DenseNet121(), DenseNet169(), DenseNet201()]

for i in Model:
    model=i

    model.trainable=False

    model.summary()

    print(str(i) + ' : ', len(model.weights))
    print(str(i) + ' : ', len(model.trainable_weights))

    time.sleep(10)

'''
vgg16
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DB9D03AF0> :  32
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DB9D03AF0> :  0

vgg19
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240
_________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DB9E31EE0> :  38
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DB9E31EE0> :  0

Xception()
Total params: 22,910,480
Trainable params: 0
Non-trainable params: 22,910,480
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DDED42B80> :  236
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022DDED42B80> :  0

ResNet101()
Total params: 44,707,176
Trainable params: 0
Non-trainable params: 44,707,176
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F76978D00> :  626
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F76978D00> :  0

ResNet101V2()
Total params: 44,675,560
Trainable params: 0
Non-trainable params: 44,675,560
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F77F52D00> :  544
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F77F52D00> :  0

ResNet152()
Total params: 60,419,944
Trainable params: 0
Non-trainable params: 60,419,944
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F7E2725B0> :  932
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F7E2725B0> :  0

ResNet152V2()
Total params: 60,380,648
Trainable params: 0
Non-trainable params: 60,380,648
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F843F6F10> :  816
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F843F6F10> :  0

ResNet50()
Total params: 25,636,712
Trainable params: 0
Non-trainable params: 25,636,712
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F85FF16A0> :  320
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F85FF16A0> :  0

ResNet50V2()
Total params: 25,613,800
Trainable params: 0
Non-trainable params: 25,613,800
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F87AADFD0> :  272
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F87AADFD0> :  0

InceptionV3()
Total params: 23,851,784
Trainable params: 0
Non-trainable params: 23,851,784
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F89DD8790> :  378
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F89DD8790> :  0

InceptionResNetV2()
Total params: 55,873,736
Trainable params: 0
Non-trainable params: 55,873,736
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F908E4AC0> :  898
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F908E4AC0> :  0

MobileNet()
Total params: 4,253,864
Trainable params: 0
Non-trainable params: 4,253,864
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F91F48A90> :  137
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F91F48A90> :  0

MobileNetV2()
Total params: 3,538,984
Trainable params: 0
Non-trainable params: 3,538,984
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F92A20880> :  262
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F92A20880> :  0

DenseNet121()
Total params: 8,062,504
Trainable params: 0
Non-trainable params: 8,062,504
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F9758EDF0> :  606
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F9758EDF0> :  0

DenseNet169()
Total params: 14,307,880
Trainable params: 0
Non-trainable params: 14,307,880
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F9C943FD0> :  846
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022F9C943FD0> :  0

DenseNet201()
Total params: 20,242,984
Trainable params: 0
Non-trainable params: 20,242,984
__________________________________________________________________________________________________
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022FA33DCC10> :  1006
<tensorflow.python.keras.engine.functional.Functional object at 0x0000022FA33DCC10> :  0

NASNetLarge()
Total params: 88,949,818
Trainable params: 0
Non-trainable params: 88,949,818
__________________________________________________________________
1546
0

NASNetMobile()
Total params: 5,326,716
Trainable params: 0
Non-trainable params: 5,326,716
_________________________________________________________________
1126
0
'''