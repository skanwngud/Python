# def sum_many(*args):
#     sum=0
#     for i in args:
#         sum=sum+i
#     return sum

# result=sum_many(1,2,3)
# print(result)


def sum_mul(choice, *args):
    if choice=='sum':
        result=0
        for i in args:
            result=+i

    elif choice=='mul':
        result=1
        for i in args:
            result=result*i

    return result

a=sum_mul(sum, 1,2,3,4)
print(a)