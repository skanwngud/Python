# 집합 연산

haystack = set(range(0, 100))
needles = set(range(0, 50, 2))

found = len(haystack & needles)
print(found)

# 집합이 아니라면
haystack = list(haystack)
needles = list(needles)

found = 0
for n in needles:
    if n in haystack:
        found += 1

print(found)

t = ()
print(type(t))

l = []
print(type(l))

d = {}
print(type(d))
