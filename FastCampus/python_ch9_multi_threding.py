"""
병렬처리

병렬: 여러개의 작업이 동시에 이루어짐 (속도 증가)

프로세스: 실행 중인 프로그램, 자원과 쓰레드로 구성
쓰레드: 프로세스 내에서 실제 작업을 수행

파이썬은 기본적으로 싱글 쓰레드에서 순차적으로 동작
(병렬처리를 하기 위해선 별도의 모듈을 불러와야한다)
"""


from threading import Thread  # 멀티 쓰레드를 위한 모듈
import time


def work(work_id, start, end, result):
    total = 0
    for i in range(start, end):
        total += i
        
    result.append(total)
    

if __name__ == "__main__":
    str_time = time.time()  # 현재 시간
    result = []
    
    th1 = Thread(target=work, args=(1, 0, 1000000, result))         # target 실행 할 함수
    th2 = Thread(target=work, args=(2, 1000001, 2000000, result))     # args 함수에 대응 되는 변수
    
    th1.start()
    th2.start()
    th1.join()  # 프로세스가 종료 될 때까지 대기, 없어도 동작은 가능 (불완전한 종료를 방지하기 위함)
    th2.join()
    
    print(result)
    
    print(time.time() - str_time)
    
"""
위의 코드를 싱글쓰레드로 실행시켜도 생각보다 실행시간이 크게 차이가 나질 않는데,
이는 파이썬의 GIL (Global Interpreter Lcok) 정책 때문.

그러나 위 정책은 cpu 작업에서만 적용 되므로 I/O 작업이 많은 병렬처리 작업에서는 효과적일 수 있다.
"""