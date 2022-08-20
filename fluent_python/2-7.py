# 레코드로 사용 된 튜플

lax_coordinates = (33.9425, -118.408056)
city, year, pop, chg, area = ('Tokyo', 2003, 32450, 0.66, 8014)
travlers_idx = [('USA', '31195855'), ('BRA', 'CE342567'), ('ESP', 'XDA205856')]

for passport in sorted(travlers_idx):
    print(passport)

for country, _ in travlers_idx:
    print(country)

print(lax_coordinates)
print(*lax_coordinates)