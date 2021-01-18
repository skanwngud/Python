import numpy as np

dataset=np.array(range(1, 11))

# 다 대 1
def split_1(dataset, size):
    x,y=list(), list()
    for i in range(len(dataset)):
        end_number=i+size
        if end_number > len(dataset)-1:
            break
        tmp_x, tmp_y=dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y=split_1(dataset, 5)
print(x,y)

# 다 대 다
def split_2(dataset, size, col):
    x,y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+size
        y_end_number=x_end_number+col
        if x_end_number > len(dataset):
            break
        tmp_x, tmp_y=dataset[i:x_end_number], dataset[x_end_number:y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

a,b=split_2(dataset, 5, 2)
print(a, b)

# 다 대 다
dataset_2=np.array([range(1, 11), range(11, 21), range(21, 31)])
dataset_2=np.transpose(dataset_2)

def split_3(dataset, size, col):
    x,y=list(), list()
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

c,d=split_3(dataset_2, 5, 1)

print(c, '\n', d)

# 다 대 다
def split_4(dataset, size, col):
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

e,f=split_4(dataset_2, 3, 2)
print(e, '\n', f)

# 다 대 다
def split_5(dataset, size, col):
    x,y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+size
        y_end_number=x_end_number+col-1
        if y_end_number > len(dataset):
            break
        tmp_x=dataset[i:x_end_number, :]
        tmp_y=dataset[x_end_number:y_end_number, :]
        x.append(x)
        y.append(y)
    return np.array(x), np.array(y)

g,h = split_5(dataset_2, 3, 1)
print(g, '\n', h)