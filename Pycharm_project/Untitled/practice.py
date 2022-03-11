import time

idx = 0

def PPrint(idx, msg):
    print(msg, 10 / idx)
    time.sleep(2)


while idx <= 10:
    try:
        PPrint(idx, "try")
        idx += 1
    except:
        idx += 1
        pass