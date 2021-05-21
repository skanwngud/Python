import cv2
import dlib
import numpy as np

# face detect
detector = dlib.get_frontal_face_detector()

# face land mark predict
predictor = dlib.shape_predictor('c:/ML/Study/Study_self/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('c:/ML/Study/Study_self/video.mp4')

All = list(range(0, 68))
right_eyebrow = list(range(17, 22))
left_eyebrow = list(range(22, 27))
right_eye = list(range(36, 42))
left_eye = list(range(42, 48))
nose = list(range(27, 36))
mouth_outline = list(range(48, 61))
mouth_inner = list(range(61, 68))
jawline = list(range(0, 17))

index = All

while True:
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)

    for face in dets:
        shape = predictor(img_frame, face)

        list_points = list()

        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                        (0, 0, 255), 3)

    cv2.imshow('result', img_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

    elif key == ord('1'):
        index == All
    elif key == ord('2'):
        index == left_eyebrow + right_eyebrow
    elif key == ord('3'):
        index == left_eye + right_eye
    elif key == ord('4'):
        index == nose
    elif key == ord('5'):
        index == mouth_outline + mouth_inner
    elif key == ord('6'):
        index == jawline

cap.release()