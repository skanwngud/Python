# namedtuple

from collections import namedtuple

city = namedtuple('City', 'name country population coordinates')

tokyo = city('Tokyo', 'JP', '36,933', ('35.689722', '139.691667'))
print(tokyo)

print(city._fields)  # city 라는 튜플이 갖는 필드들

LatLong = namedtuple('LatLong', 'lat long')
delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77208889))
delhi = city._make(delhi_data)  # 반복형 객체 (iterable 한 list, tuple 등) 로부터 namedtuple 을 만든다 City(*delhi_data) 와 같음

print(delhi)

print(delhi._asdict())  # dictionary 로 변환