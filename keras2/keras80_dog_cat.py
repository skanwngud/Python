# 이미지는 c:/data/image/vgg/ 에 4개 넣을 것
# file name: dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

# 라이브러리 임포트
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions # 복호화 (해석)


# 이미지 불러오기
img_dog=load_img('c:/data/image/vgg/dog1.jpg', target_size=(224, 224)) # 케라스에서 불러옴 / <PIL.Image.Image image mode=RGB size=224x224 at 0x148A55085B0>
img_cat=load_img('c:/data/image/vgg/cat1.jpg', target_size=(224, 224))
img_lion=load_img('c:/data/image/vgg/lion1.jpg', target_size=(224, 224))
img_suit=load_img('c:/data/image/vgg/suit1.jpg', target_size=(224, 224))

# 시각화
# plt.imshow(img_lion)
# plt.show()

# print(img_suit)

# 이미지의 수치화
arr_dog=img_to_array(img_dog)
arr_cat=img_to_array(img_cat)
arr_lion=img_to_array(img_lion)
arr_suit=img_to_array(img_suit)

# print(arr_suit) # RGB 형태
# print(type(arr_dog)) # <class 'numpy.ndarray'>
# print(arr_dog.shape) # (224, 224, 3)

# 위의 이미지는 RGB 이며, VGG16 에 들어갈 때에는 BGR 로 되어야한다
# preprocess_input 처리

arr_dog=preprocess_input(arr_dog)
arr_cat=preprocess_input(arr_cat)
arr_lion=preprocess_input(arr_lion)
arr_suit=preprocess_input(arr_suit)

# print(arr_suit)
# print(type(arr_dog))
# print(arr_dog.shape)

arr_input=np.stack(
    [arr_dog, arr_cat, arr_lion, arr_suit]
) # np.stack 찾아볼 것

print(arr_input.shape) # (4, 224, 224, 3)

# 모델구성
model=VGG16()

results=model.predict(arr_input)

print(results)
print('results.shape : ', results.shape)

'''
[[1.47623949e-08 2.11909623e-09 1.76055028e-08 ... 1.33881639e-09
  2.32005689e-07 1.17042964e-06]
 [7.25995832e-08 3.83385810e-07 1.03320463e-06 ... 1.08508864e-07
  2.27376149e-05 2.98894563e-04]
 [1.32918819e-06 2.08956044e-05 1.69381235e-06 ... 1.71409226e-06
  1.56086790e-05 1.14980270e-03]
 [1.94402355e-06 1.89213793e-07 4.66124447e-07 ... 1.40911780e-07
  3.66867857e-06 4.14425522e-05]]
results.shape :  (4, 1000)
'''

# 이미지 결과 확인
decode_results=decode_predictions(
    results
)

print('='*50)
print('results[0] : ', decode_results[0])
# results[0] :  [('n02099601', 'golden_retriever', 0.45978424), ('n02104029', 'kuvasz', 0.26541054), ('n02111500', 'Great_Pyrenees', 0.21658602), ('n02090721', 'Irish_wolfhound', 0.008457371), ('n02106662', 'German_shepherd', 0.007232256)]
# 각 항목별로 해당 이미지일 확률을 의미한다
print('='*50)
print('results[1] : ', decode_results[1])
# results[1] :  [('n02123045', 'tabby', 0.61105937), ('n02124075', 'Egyptian_cat', 0.1921394), ('n02123159', 'tiger_cat', 0.15950352), ('n02971356', 'carton', 0.0032448815), ('n04265275', 'space_heater', 0.0026122732)]
print('='*50)
print('results[2] : ', decode_results[2])
# results[2] :  [('n03291819', 'envelope', 0.21738482), ('n02786058', 'Band_Aid', 0.091208614), ('n03598930', 'jigsaw_puzzle', 0.06482566), ('n03908618', 'pencil_box', 0.058071256), ('n06359193', 'web_site', 0.05694257)] 
print('='*50)
print('results[3] : ', decode_results[3])
# results[3] :  [('n04350905', 'suit', 0.66315013), ('n03680355', 'Loafer', 0.10518959), ('n04591157', 'Windsor_tie', 0.08765824), ('n02883205', 'bow_tie', 0.04615856), ('n03832673', 'notebook', 0.03443192)]
print('='*50)

