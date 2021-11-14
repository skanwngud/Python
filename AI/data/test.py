import os
import cv2 as cv

vid_path = "D:/super_shot/"
path = "D:/super_shot/cap_frame/"

file_list = os.listdir(vid_path)

for file in file_list:
    try:
        if file.split(sep=".")[1] != "mp4":
                file_list.remove(file)
    except:
        print("no file")

for file in file_list:
    count = 1
    result = True
    try:
        os.makedirs(path + f"{file}")
    except:
        pass
    vidcap = cv.VideoCapture(vid_path + file)
    print(vidcap.isOpened())
    while result:
        try:
            _, frame = vidcap.read()
            cv.imwrite(path + f"{file}/{count}.jpg", frame)
            print(f"{file} - {count}.jpg save completed")

            if cv.waitKey(10) == 27:
                print("stop")
                break
        except:
            print("save done!")
            break
        count += 1