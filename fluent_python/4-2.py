# bytes, bytearray 로 저장한 5바이트 시퀀스

cafe = bytes('café', encoding='utf-8')
print(cafe)

print(cafe[0])
print(cafe[:1])

cafe_arr = bytearray(cafe)
print(cafe_arr)

print(cafe_arr[-1:])
