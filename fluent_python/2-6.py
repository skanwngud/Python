# 제네레이터 표현식을 이용한 데카르트 곱

colors = ["white", "black"]
sizes = ["S", "M", "L"]

for tshirt in (f"{c}, {s}" for c in colors for s in sizes):
      print(tshirt)

# 제네레이터는 6개의 요소를 갖는 배열이나 튜플 등을 만들지 않고 한 번에 하나씩의 값만 리턴한다
