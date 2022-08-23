# 길이가 3 개인 리스트

board = [['_'] * 3 for i in range(3)]  # 하나의 리스트에 하나의 참조값을 가진다
print(board)

board[1][2] = 'X'
print(board)

weird_board = [['_'] * 3] * 3  # 동일한 리스트에 여러개의 참조값을 가진다
print(weird_board)

weird_board[1][2] = 'O'
print(weird_board)

#######
# board 의 실질적인 작동원리
board = []
for i in range(3):
    row = ['_'] * 3
    board.append(row)  # 추가 될 때마다 새롭게 객체가 생성 됨

#######
# weird_board 의 실질적인 작동원리
row = ['_'] * 3
board = []
for i in range(3):
    board.append(row)  # 동일한 행이 3번 추가 된