import numpy as np

a=np.array([ [ 1, 2, 3 ] , [ 4, 5, 6 ] ]) # (2,3)
print(a.shape)

b=np.array([[1],[2],[3],[4]]) # (4,1)
c=np.array([[1,2],[3,4],[5,6]]) # (3,2)

print(b.shape)
print(c.shape)

d=np.array([[[[[1,2,3],
               [4,5,6]]],

             [[[7,8,9],
               [10,11,12]]]],
               
            [[[[13,14,15],
               [16,17,18]]],

             [[[19,20,21],
               [22,23,24]]]]]) # (2,2,1,2,3)

print(d.shape)

e=np.array([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]]]) # (2,2,2,3)
print(e.shape)