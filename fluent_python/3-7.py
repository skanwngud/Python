# UserDict 예제

import collections


class StrKeyDict0(collections.UserDict):  # UserDict 를 상속
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):  # 저장 된 키가 모두 str 형이므로 self.data 에서 바로 조회 가능
        return str(key) in self.data

    def __setitem__(self, key, value):  # 모든 키를 str 형으로 반환한다다
       self.data[str(key)] = value

