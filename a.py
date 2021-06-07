list = [1, 2, 1, 4]

for i in range(4):
    for j in range(4):
        for k in range(len(list)):
            if j - list[k] == 0:
                print('*')