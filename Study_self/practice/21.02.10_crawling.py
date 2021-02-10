from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus

baseUrl="https://pixabay.com/ko/photos/search/"
plusUrl=input('검색어 :')
crawl_num=int(input('크롤링 할 갯수 : '))

url=baseUrl+quote_plus(plusUrl+'/')
html=urlopen(url)
soup=bs(html, 'html.parser')
img=soup.find_all(class_='no_header_seach')

n=1
for i in img:
    print(n)
    imgUrl=i['data-source']
    with urlopen(imgUrl) as f:
        with open('c:/data/image/project/img' + str(n) + '.jpg', 'wb') as h:
            img=f.read()
            h.write(img)
    n += 1
    if n > crawl_num:
        break

print('이미지 크롤링 종료')
