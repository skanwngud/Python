import numpy as np

dataset=np.array([range(1,11), range(11,21), range(21,31)])
dataset=np.transpose(dataset)

def split(dataset, size, col):
    x, y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+size
        y_end_number=x_end_number+col-1
        if y_end_number > len(dataset):
            break
        tmp_x=dataset[i:x_end_number, :-1]
        tmp_y=dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y=split(dataset, 3, 2)

print('x:\n', x)
print('y:\n', y)