# 지능형 리스트 (list comp) 를 이용한 데카르트 곱
# 데카르트 곱; 2 개 이상의 리스트의 모든 항목을 이용해서 만든 튜플로 구성 된 리스트

rank = ['A', 'K', 'Q']
speicies = ['spade', 'heart', 'diamond', 'clover']

rs = [(x, y)
      for x in rank
      for y in speicies]
print(rs)
