# encoding, decoding

s = 'café'
print(len(s))

b = s.encode('utf8')
print(b)
print(len(b))

d = b.decode('utf8')
print(d)
print(len(d))
