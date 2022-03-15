import time


def test(idx):
    res = 0

    try:
        res = 10 // idx
        if res == 0:
            raise ValueError


    except ZeroDivisionError:
        time.sleep(1)
        return

    except ValueError:
        time.sleep(1)
        return

    time.sleep(1)

    return res


idx = 0

# while idx <= 100:
#     print(test(idx))
#     idx += 1


import importlib.metadata as met

