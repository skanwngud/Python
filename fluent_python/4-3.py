# 배열의 원시 데이터에서 bytes 로 초기화하기

import array
numbers = array.array('h', [-2, -1, 0, 1, 2])
octets = bytes(numbers)
print(octets)
