# 지능형 딕셔너리 (딕셔너리 컴프리헨션)

DIAL_CODES = [
    (86, 'China'),
    (91, 'India'),
    (1, 'United States'),
    (62, 'Indonesia'),
    (55, 'Brazil'),
    (880, 'Bangladesh'),
    (234, 'Nigeria'),
    (7, 'Russia'),
    (81, 'Japan'),
    (82, 'Korea')
]

country_code = {country: code for code, country in DIAL_CODES}  # 튜플의 언팩킹을 이용하여 키, 밸류 값으로 나눔
print(country_code)

print({code: country.upper() for country, code in country_code.items()})  # dict.itmes() 함수를 이용하여 언팩킹 후 재할당

country_code.update({'Korea': 11})

print(country_code)