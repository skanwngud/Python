# dict 와 set 의 성능 실험

var = 1000
haystack = list(range(0, var))
needles = list(range(0, var * 2, 2))

print(haystack)
print(needles)

found = 0
for n in needles:
    if n in haystack:
        found += 1

print(found)

haystack = {key: "" for key in range(0, var)}
