import numpy as np

dataset=np.array([range(1,11), range(11,21), range(21,31)])
dataset=np.transpose(dataset)

def split(data, x_low, x_col, y_low, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+x_low
        y_end_number=x_end_number+y_low-1
        if y_end_number > len(data):
            break
        tem_x=data[i:x_end_number, x_col]
        tem_y=data[x_end_number-1:y_end_number, y_col]
        x.append(tem_x)
        y.append(tem_y)
    return np.array(x), np.array(y)

x,y=split(dataset, 6,0,1,0)

print(x, '\n', y)
print(x.shape)
print(y.shape)