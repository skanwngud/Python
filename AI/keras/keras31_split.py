import numpy as np

a=np.array(range(1,11))
size=5

# def split_x(seq, size):
#     aaa=[]
#     for i in range(len(seq)-size+1):
#         subset=seq[i:(i+size)]
#         aaa.append([item for item in subset])
#     print(type(aaa))
#     return np.array(aaa)

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset=split_x(a, size)
print('='*30)
print(dataset)
print(a.shape) # (10, )
print(dataset.shape) # (6, 5)

'''
def split_x(seq, size):             seq, size 를 변수로 갖는 slpit_x 라는 함수를 선언
    aaa=[]              aaa 는 리스트 (함수를 통과하며 나오는 값들을 저장)
    for i in range(len(seq)-size+1):                변수 i 가 1~10 의 범위 안에 있는동안 
        subset=seq[i:(i+size)]              subset 는 i 와 i + size 의 범위동안 연속적으로 값을 저장
        aaa.append([item for item in subset])               subset 안에 item 이 있는 동안 item 을 aaa 리스트에 추가
    print(type(aaa))                aaa 의 타입을 출력
    return np.array(aaa)                np.array(aaa) 를 반환

size=5
<class 'list'>
==============================
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

size=3
<class 'list'>
==============================
[[ 1  2  3]
 [ 3  4  5]
 [ 4  5  6]
 [ 5  6  7]
 [ 6  7  8]
 [ 7  8  9]
 [ 8  9 10]]
'''

