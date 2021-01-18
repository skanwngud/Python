import numpy as np

x=np.array(range(1, 11))

def split(data, size, col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+size
        y_end_number=x_end_number+col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number]
        tmp_y=data[x_end_number:y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y=split(x, 5, 2)
print(x, '\n', y)
print(x.shape) # (4, 5)
print(y.shape) # (4, 2)
