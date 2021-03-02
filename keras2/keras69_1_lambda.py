gradient=lambda x:2*x-4 # x에 2x-4 를 넣었을 때 특정 값을 리턴해줌

def gradient2(x):
    temp=2*x-4
    return temp # 위의 gradient 를 풀어 씀

x=3

print(gradient(x)) # 2
print(gradient2(x)) # 2