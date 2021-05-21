# open CV import
import cv2

# define function
# cascade 를 사용하기 위해선 gray scale 로 바꿔줘야 하므로 cvtColor 를 사용한다
def videoDetector(cam, cascade):
    while True:
        ret, img = cam.read()
        img = cv2.resize(img, dsize = None, fx=0.75, fy=0.75)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect face
        results = cascade.detectMultiScale(
            gray,
            scaleFactor = 1.5, # 객체탐지를 위해 이미지를 다양한 크기로 변화시키는데 이 때의 변화율
            minNeighbors = 5,
            minSize = (20, 20)
        )

        for box in results:
            x,y,w,h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), thickness=2)

        cv2.imshow('facenet', img)
        if cv2.waitKey(1) > 0:
            break

def imgDetector(img, cascade):
    img = cv2.resize(img, dsize = None, fx = 0.5, fy = 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face
    results = cascade.detectMultiScale(
        gray,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (20, 20)
    )

    for box in results:
        x,y,w,h = box
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), thickness=2)

    cv2.imshow('facenet', img)
    cv2.waitKey(10000)

# pretrained data load
cascade_filename = 'c:/ML/Study/Study_self/haarcascade_frontalface_alt.xml'

# load cascade classifier model
cascade = cv2.CascadeClassifier(cascade_filename)

# video load
cam = cv2.VideoCapture('c:/ML/Study/Study_self/video.mp4')

img = cv2.imread('c:/ML/Study/Study_self/kakao.jpg')

img_de = imgDetector(img, cascade)
# videoDetector(cam, cascade)
