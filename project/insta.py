

from urllib.request import urlopen
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import matplotlib.pyplot as plt
import time
import requests
import shutil
import os
import re
headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"
}
username = 'juhyeong_y'
userpw = 'Ahdwl0312!'
crawling_count=100

def select_first(driver):
    first = driver.find_element_by_css_selector('div._9AhH0') 
    #find_element_by_css_selector 함수를 사용해 요소 찾기
    first.click()
    time.sleep(3) #로딩을 위해 3초 대기
    
def insta_searching(word):  #word라는 매개변수를 받는 insta_searching 이라는 함수 생성
    url = 'https://www.instagram.com/explore/tags/' + word
    return url

def get_content(driver):
    # 1. 현재 페이지의 HTML 정보 가져오기
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')    
    # 2. 본문 내용 가져오기
    try:  			#여러 태그중 첫번째([0]) 태그를 선택  
        content = soup.select('div.C4VMK > span')[0].text 
        			#첫 게시글 본문 내용이 <div class="C4VMK"> 임을 알 수 있다.
                                #태그명이 div, class명이 C4VMK인 태그 아래에 있는 span 태그를 모두 선택.
    except:
        content = ' ' 
    # 3. 본문 내용에서 해시태그 가져오기(정규표현식 활용)
    tags = re.findall(r'#[^\s#,\\]+', content) # content 변수의 본문 내용 중 #으로 시작하며, #뒤에 연속된 문자(공백이나 #, \ 기호가 아닌 경우)를 모두 찾아 tags 변수에 저장
    # 4. 작성 일자 가져오기
    try:
        date = soup.select('time._1o9PC.Nzb55')[0]['datetime'][:10] #앞에서부터 10자리 글자
    except:
        date = ''
    # 5. 좋아요 수 가져오기
    try:
        like = soup.select('div.Nm9Fw > button')[0].text[4:-1] 
    except:
        like = 0
    # 6. 위치 정보 가져오기
    try:
        place = soup.select('div.JF9hh')[0].text
    except:
        place = ''
    try:
        img_url=soup.find('div',class_='KL4Bh').find('img').get('src')
    except:
        img_url = ''
    data = [content, tags, img_url]
    return data 

def move_next(driver):
    right = driver.find_element_by_css_selector('a._65Bje.coreSpriteRightPaginationArrow') 
    right.click()
    time.sleep(3)


from selenium import webdriver
from bs4 import BeautifulSoup
import time
import re 
#1. 크롬으로 인스타그램 - '사당맛집' 검색
driver = webdriver.Chrome('C:/Study/project/chromedriver.exe')
word = '마스크'
url = insta_searching(word)
driver.get(url) 
time.sleep(4) 
#2. 로그인 하기


driver.find_element_by_name('username').send_keys(username)
driver.find_element_by_name('password').send_keys(userpw)
driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[3]/button/div').click()
time.sleep(5)
driver.find_element_by_xpath('//*[@id="react-root"]/section/main/div/div/div/section/div/button').click()
time.sleep(5)
driver.find_element_by_xpath('/html/body/div[4]/div/div/div/div[3]/button[2]').click()
time.sleep(5)

time.sleep(1) 
driver.get(url)
time.sleep(4) 
select_first(driver) 
results = [] 
html = driver.page_source
soup = BeautifulSoup(html, 'lxml') 
content = soup.select('div.C4VMK > span')[0].text
tags = re.findall(r'#[^\s#,\\]+', content)
a = soup.find_all('div',class_='KL4Bh')[0].find('img').get('src')
n=0
tags=[k.replace("#",'') for k in tags]
resp=requests.get(a,stream=True)
filename='img2/'+'test'+'.jpg'
local_file = open('img2/'+'test'+'.jpg','wb')
resp.raw.decode_content=True
shutil.copyfileobj(resp.raw,local_file)
k = plt.imread(filename)
print(n)
plt.imshow(k)
