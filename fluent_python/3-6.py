# StrKeyDict0 class 선언
# 사용자 정의 매핑형을 만들 땐 dict 보다 collections.UserDict 클래스를 상속하는 것이 더 낫다.

# 조회 할 때 문자열로 변환해주는 dict
class StrKeyDict0(dict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()

# 그 외의 매핑형
import collections

od = collections.OrderedDict()
# 키를 삽입한 순서대로 유지하여 반복하는 순서를 예측 할 수 있음.
# popitem(), popitem(last=True) 등으로 가장 최근 혹은 가장 처음에 삽입한 항목을 꺼냄.

cm = collections.ChainMap()
# 매핑의 목록을 담고 있으며 한꺼번에 모두 검색할 수 있다.
# 각 매핍을 차례대로 검색하고 그 중에 하나라도 키가 검색 되면 성공한다.

ud = collections.UserDict()
# 표준 dict 처럼 작동하는 매핑을 순수 파이썬으로 구현한 클래스
# 상속해서 사용하도록 설계 되어있다