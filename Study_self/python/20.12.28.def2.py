# def sum_many(*args):
#     sum=0
#     for i in args:
#         sum=sum+i
#     return sum

# result=sum_many(1,2,3)
# print(result)


# def a(choice, *args):
#     if choice=='sum':
#         result=0
#         for i in args:
#             result=+i

#     elif choice=='mul':
#         result=1
#         for i in args:
#             result=result*i

#     return result

# b=a(sum, 1,2,3,4)
# print(b)

nick='stupid'

def say_nick(nick):
    if nick=='stupid':
        return
    print('나의 별명은 %s입니다.'% nick)

