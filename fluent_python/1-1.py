# 파이썬 카드 한 벌
# __getitem__(), __len__() 을 이용

import collections

Card = collections.namedtuple("Card", ["rank", "suit"])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = "spades diamonds, clubs, hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

# 지정 카드 뽑기
base_card = Card(7, "hearts")
print(base_card)

# 무작위 카드 뽑기
deck = FrenchDeck()
print(deck[0])

# random 을 이용한 무작위 카드를 랜덤하게 뽑기
from random import choice
print(choice(deck))

# __contains__ 메서드를 이용한 in 연산
print(Card('7', 'spades') in deck)
print(Card('9', 'beasts') in deck)

# 카드 정렬하기
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

