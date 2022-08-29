# setdefault

"""단어가 나타나는 위치를 가리키는 인덱스를 만든다."""

import sys
import re

WORD_RE = re.compile(r'\w+')
index = {}

with open(sys.argv[0], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start()+1
            location = (line_no, column_no)

            # 보기 좋은 코드는 아니지만 설명하기 위해 이렇게 구현했다. - by, writer.
            occurences = index.get(word, [])  # 단어에 대한 새로운 occurences 를 가져오거나 단어가 없으면 빈 배열을 가져온다
            occurences.append(location)  # 새로 만든 location 에 occurences 를 추가한다
            index[word] = occurences  # 뼌경 된 occurences 를 딕셔너리에 넣는다. 그 후 한 번 더 index 를 검색한다.

for word in sorted(index, key=str.upper):  # str.upper() 를 호출하지 않고 이 함수에 대한 참조만 전달해서 sorted 를 정규화하게 만든다. -> 일급함수를 사용하는 하나의 예
    print(word, index[word])


##########

"""위의 코드를 setdefualt 를 이용하여 좀 더 깔끔하게 만들었다"""
import sys
import re

WORD_RE = re.compile(r'\w+')
index = {}

with open(sys.argv[0], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start()+1
            location = (line_no, column_no)
            index.setdefault(word, []).append(location)  # setdefault() 를 사용하면 한 번 더 검색 할 필요가 없다.

for word in sorted(index, key=str.upper):
    print(word, index[word])

"""
# setdefault 의 기본 
my_dict.setdefault(key, []).append(new_value)

# setdefault 의 작동 알고리즘 (물론 setdefault 가 훨씬 우월하다)
if key not in my_dict:
    my_dict[key] = []
my_dict[key].append(new_value)
"""
