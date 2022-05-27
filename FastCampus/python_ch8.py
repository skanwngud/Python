# 직렬화
# 객체를 연속적인 데이터로 변환한다 (Serialize)

# pickle
# pickle.dump(출력 객체, 파일 객체)

import pickle

with open('test.txt', 'wb') as f:
    pickle.dump([1, 2, 3, 4], f)

with open('test.txt', 'rb') as f:
    byte_file = pickle.load(f)
    
print(byte_file)