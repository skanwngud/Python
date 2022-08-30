# defaultdict: 존재하지 않는 키로 검색할 때 요청에 따라 항목을 생성함

import sys
import re
import collections

WORD_RE = re.compile(r'\w+')

index = collections.defaultdict(list)  # list 생성자를 갖고있는 defaultdict 를 생성한다.
with open(sys.argv[0], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start()+1
            location = (line_no, column_no)
            index[word].append(location)
            # word 가 index 에 들어있지 않으면 리스트를 생성해서 index[word] 에 할당한 후 반환하므로 append(location) 은 항상 성공
            # word 가 존재하지 않으면 index[word] 는 리스트가 되므로 list.append(location) 연산이 항상 성공한다
            # 그 후 index[word] 에 location 값이 들어감

for word in sorted(index, key=str.upper):
    print(word, index[word])
