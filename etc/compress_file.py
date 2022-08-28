# 파일 압축

import zipfile
import os

path = ""
file_list = os.listdir(path)

for file in file_list:
    print(path + file)
    img_list = os.listdir(path + file)
    for img in img_list:
        print(path + file + img)
        zfile = zipfile.ZipFile(path + file + img, "w")
        zfile.write(img)
zfile.close()