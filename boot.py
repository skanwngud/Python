import sys

ep = sys.float_info.epsilon

a = 9.25
diff = (2**3)*ep

print(diff)

b = a + diff
print(b)

a = (2.0)**53
print(a)

b = a + 1.0

print(a==b)