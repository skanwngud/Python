# memory view

import array
numbers = array.array('h', [-2, -1, 0, 1, 2])  # 짧은 정수 코드 'h'
mbmv = memoryview(numbers)  # memory view 도 5개의 항목을 동일하게 본다
print(len(mbmv))
print(mbmv[0])

mbmv_oct = mbmv.cast('B')  # Unsigned char code 'B' 로 형변환 한 mbmv_oct
print(mbmv_oct.tolist())  # 리스트로 변환하여 값 확인

mbmv_oct[5] = 4
print(mbmv_oct)
print(numbers)  # unsigned int sequence 에서 최상위 바이트에서 4 는 1024

