# 파일 압축

import zipfile
import shutil
import os

path = "path/to/dir"
file_list = os.listdir(path)

for file in file_list:
    print(file)
    shutil.make_archive(path + file, "zip", path + file)

