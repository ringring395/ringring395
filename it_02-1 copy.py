#웹크롤링
# selenium, BeautifulSoup 설치
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager    #가상의 웹브라우저로 옮겨서 읽음
from selenium.webdriver.common.by import By     #셀레니움 4버전에선 추가해줌
from bs4 import BeautifulSoup          # js파일은 못읽음 -> 셀레니움 도움 필요
import re

# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"

# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
driver = webdriver.Chrome(ChromeDriverManager().install())  # for Windows
driver.get(source_url)
# 브라우저 실행 시간을 설정(바로 종료되는거 방지)
driver.implicitly_wait(10)
# 필요한 태그를 찾아서 들어가자.
table_rows = driver.find_elements(By.XPATH,'//*[@id="C6Rc9QlVe"]/div[2]/div/div/div/div/div/article/div[3]/div/div/div/div[1]/div/div/table/tbody/tr/td/a')
page_urls = []
for i in range(0,len(table_rows)):
    page_urls.append(table_rows[i].get_attribute("href"))   # a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성합니다.

# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
for page in page_urls[:3]:
    print(page)

# 크롤링에 사용한 브라우저를 종료합니다.
driver.close()