from multiprocessing import Pool, Process  # 멀티 프로세싱 관련 모듈
import os


def f(x):
    return x * x

print(__name__)  # __mp__main__

if __name__ == "__main__":
    p = Pool(4)  # 서브 프로세스 갯수 생성
                 # 서브 프로세스 갯수를 많이 띄우게 되면 오히려 효율이 떨어져 속도가 더 느려질 수 있다
                 # 당시 자원과 상황에 따라 적당히 조절해줘야함
    result = p.map(f, range(100))
    p.close()    # 프로세스 종료
    
    print(result)
    
    numbers = [1, 2, 3, 4]
    proc1 = Process(target=f, args=(numbers[0],))
    proc1.start()
    proc2 = Process(target=f, args=(numbers[1],))
    proc2.start()
    proc1.close()
    proc2.close()
    
"""
Pool; 처리 할 작업들을 Pool 에 뿌려놓고 임의로 병렬처리를 진행
Process; 각 프로세스 별로 처리 할 작업들을 명시적으로 할당하고 처리 진행

실제의 차이는 좀 더 복잡함
"""