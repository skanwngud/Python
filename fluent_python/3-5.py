# __missing__ method
# 기본형태의 dict 에서는 __missing__() 메소드는 존재하지 않지만 __getitem__() 메소드를 이용할 때 호출할 수 있다.
# 따라서 __getitem__() 메소드를 이용하여 키를 찾을 때, 해당 dict 에 키가 존재하지 않아도 KeyError 가 발생하지 않는다.

d = {
    "2": "two",
    "4": "four"
}

print(d["2"])
print(d["4"])
# print(d["1"])  # KeyError 발생

print(d.get('2'))
print(d.get('4'))
print(d.get('1', None))  # KeyError 발생하지 않고 None 값을 뱉음

print("2" in d)
print("4" in d)
print("1" in d)
