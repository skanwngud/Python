import cv2 as cv
import numpy as np

# 기본 이미지 저장 path
path = 'C:/Users/yjh/Downloads/'

img = cv.imread(path + '4692.jpg') # 이미지 로드
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지 흑백 변환
ret, dst = cv.threshold(img2, 200, 255, cv.THRESH_BINARY) # 이진화



# image show
cv.imshow("img", img)
cv.imshow("img2", img2)
cv.imshow("img3", dst)
cv.waitKey(0)