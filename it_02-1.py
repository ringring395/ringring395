#웹 크롤링_웹 사이트의 url 주소 정보 추출
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re

#크롬 실행 경로
chrome_path = "chromedriver.exe"
#크롤링할 사이트 주소 정의(나무위키_최근변경내역)
source_url = "https://namu.wiki/RecentChanges"
#크롬을 이용해 웹 크롤링
driver = webdriver.Chrome(executable_path = chrome_path)
#페이지 소스
driver.get(source_url)
req = driver.page_source
#html 구조에 기반해서 데이터 파싱
soup = BeautifulSoup(req, "html.parser")
content_table = soup.find(name="table")
table_body = content_table.find(name="tbody")
table_row = table_body.find_all(name="tr")
#a태그 href속성을 리스트로 추출 -> 크롤링할 페이지 리스트 생성
page_url_base = "https://namu.wiki"
page_urls = []
for index in range(0, len(table_row)) :
    first_td = table_row[index].find_all("td")[0]
    td_url = first_td.find_all("a")
    if len(td_url) > 0 :
        page_url = page_url_base + td_url[0].get("href")
        if "png" not in page_url :
            page_urls.append(page_url)
#중복 url 제거
page_urls = list(set(page_urls))
for page in page_urls[:3] :
    print(page)
