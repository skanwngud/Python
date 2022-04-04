import math
import numpy as np

v = 55
seta = 14

distance = round((math.pow(v, 2) * np.abs(math.sin(2 * seta))) / 9.8, 2)


def solution(numbers):
    answer = 0
    numbers = sorted(numbers)
    for idx in range(len(numbers)):
        if len(numbers[:idx]) <= numbers[idx] <= len(numbers[idx:]):
            answer = numbers[idx]
    return answer


if __name__ == "__main__":
    a = [4, 1, 0, 4, 5]
    print(solution(a))

    b = []
    c = ()
    print(b)
    print(c)

    print(len(b))
    print(len(c))

    if len(b) == 0 or len(c) == 0:
        print('d')
