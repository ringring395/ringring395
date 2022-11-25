#101p
# 최근 변경 내역 페이지 키워드 분석
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import re
#키워드 분석
from konlpy.tag import Okt
from collections import Counter
#시각화
import random
import pytagcloud
import webbrowser
from IPython.display import Image

# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"
# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(source_url)
driver.implicitly_wait(10)
#html 구조 분석(필요한 데이터가 있는 곳까지 찾아가야함)
table_rows = driver.find_elements(By.XPATH,'//*[@id="C6Rc9QlVe"]/div[2]/div/div/div/div/div/article/div[3]/div/div/div/div[1]/div/div/table/tbody/tr')
print(table_rows)
page_urls = []
#for i in range(0,len(table_rows)): # 시간이 너무 오래 걸림
# 어차피 맨 밑에 상위 다섯번째까지만 출력하면 되니 5번만 반복.
for i in range(0,5):    
    first_td=table_rows[i].find_elements(By.TAG_NAME,"td")
    td_url=first_td[0].find_elements(By.TAG_NAME,"a")
    if len(td_url) >0 :
        page_url=td_url[0].get_attribute("href")
        # a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성
        print(page_url)
        if 'png' not in page_urls:
            page_urls.append(page_url)
# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
print(page_urls)

columns = ['title','category','content_text']
df=pd.DataFrame(columns=columns)

for page_url in page_urls:
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(10)
    driver.get(page_url)
    req=driver.page_source
    soup=BeautifulSoup(req,"html.parser")
    contents_table=soup.find(attrs={"class":"SVuqC-pU"})
    title=contents_table.find_all("h1")[0]

    if len(contents_table.find_all("ul"))>0:
        category=contents_table.find_all("ul")[0]
    else:
        category=None

    content_paragraphs=contents_table.find_all(name="div", attrs={"class":"UtCm7-qJ"})
    content_corpus_list=[]

    if title is not None:
        row_title=title.text.replace("\n"," ")
    else:
        row_title=""
    
    if content_paragraphs is not None:
        for paragraghs in content_paragraphs:
            if paragraghs is not None:
                content_corpus_list.append(paragraghs.text.replace("\n"," "))
            else:
                content_corpus_list.append("")
    else:
        content_corpus_list("")

    if category is not None:
        row_category=category.text.replace("\n"," ")
    else:
        row_category=""

    row=[row_title,row_category,"".join(content_corpus_list)]
    series=pd.Series(row, index=df.columns)
    df=df.append(series, ignore_index=True)

    driver.close()

#정규식 표현
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글만 
    result = hangul.sub('', text)
    return result
# 전처리
df["title"]=df["title"].apply(lambda x:text_cleaning(x))
df["category"]=df["category"].apply(lambda x:text_cleaning(x))
df["content_text"]=df["content_text"].apply(lambda x:text_cleaning(x))
# 한글은 다 합침
title_corpus="".join(df["title"].tolist())
category_corpus="".join(df["category"].tolist())
content_corpus="".join(df["content_text"].tolist())

print("---title---")
print(title_corpus)
print("---category---")
print(category_corpus)
print("---content---")
print(content_corpus)

nouns_tagger = Okt()
nouns = nouns_tagger.nouns(content_corpus)
count = Counter(nouns)
print("---형태소 단위 추출---")
print(count)
#한글자 키워드는 삭제함
remove_char_counter=Counter({x : count[x] for x in count if len(x)>1})
print(remove_char_counter)
#키워드 가다듬기
#불용어 사전
korean_stopwords_path = "korean_stopwords.txt"
#텍스트 파일 오픈
with open(korean_stopwords_path, encoding='utf-8') as f :
    stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]
print(stopwords[:10])
#나무위키 페이지에 적용 필요한 불용어 추가
namu_wiki_stopwords = ["상위", "문서", "내용", "누설", "아래", "해당",
                    "설명", "표기", "추가", "모든", "사용", "매우", "가장",
                    "줄거리", "요소", "상황", "편집", "틀", "경우", "때문",
                    "모습", "정도", "이후", "사실", "생각", "인물", "이름", 
                    "년월"]
for stopword in namu_wiki_stopwords :
    stopwords.append(stopword)
#추출한 데이터에서 불용어 제거
remove_char_counter = Counter({x : remove_char_counter[x] for x in count if x not in stopwords})
print(remove_char_counter)
#시각화
#빈도수가 높은 40개의 단어를 선정
ranked_tags = remove_char_counter.most_common(40)
#출력할 40개 단어 입력(최대 크기 제한 : 80)
taglist = pytagcloud.make_tags(ranked_tags, maxsize=80)
#이미지 생성(폰트 : 나눔고딕)
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600),
                        fontname='NanumGothic', rectangular=False)
#출력하기
Image(filename="wordcloud.jpg")