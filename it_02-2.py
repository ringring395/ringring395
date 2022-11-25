from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re

# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/w/2022%20FIFA%20%EC%9B%94%EB%93%9C%EC%BB%B5%20%EC%B9%B4%ED%83%80%EB%A5%B4/%EC%A4%91%EA%B3%84/%EA%B5%AD%EB%82%B4%20%EC%9D%BC%EC%A0%95"
# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
driver = webdriver.Chrome(ChromeDriverManager().install())  # for Windows

driver.get(source_url)
req=driver.page_source
soup=BeautifulSoup(req, "html.parser")
contents_table=soup.find(name="div", attrs={"class":"SVuqC-pU"})
title=contents_table.find_all("h1")[0]      # 제목
category=contents_table.find_all("ul")[0]   # 카테고리(분류)
content_paragraphs=contents_table.find_all(name="div", attrs={"class":"UtCm7-qJ"})
content_corpus_list=[]          #본문 내용

for paragraphs in content_paragraphs:
    content_corpus_list.append(paragraphs.text)
content_corpus = "".join(content_corpus_list)

print(title.text)       #페이지 전체 제목
print("\n")
print(category.text)    #페이지 카테고리(분류) 제목
print("\n")
print(content_corpus)   #페이지 내용

driver.close()