#감성 분류 : 맛집 리뷰
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

import re
import time

#236p
#카카오맵 크롤링
# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://map.kakao.com/"
# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
driver = webdriver.Chrome(ChromeDriverManager().install())
#카카오 지도에 접속
driver.get(source_url)      #페이지 소스를 파이썬으로 가져옴
# 검색창에 검색어를 입력합니다
searchbox = driver.find_element(By.XPATH, "//input[@id='search.keyword.query']")
searchbox.send_keys("강남역 고기집")
# 검색버튼을 눌러서 결과를 가져옵니다
searchbutton = driver.find_element(By.XPATH,"//button[@id='search.keyword.submit']")
driver.execute_script("arguments[0].click();", searchbutton)
# 검색 결과를 가져올 시간을 기다립니다
time.sleep(2)
# 검색 결과의 페이지 소스를 가져옵니다
html = driver.page_source
# BeautifulSoup을 이용하여 html 정보를 파싱합니다
soup = BeautifulSoup(html, "html.parser")
moreviews = soup.find_all(name="a", attrs={"class":"moreview"})
# a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성합니다.
page_urls = []
for moreview in moreviews:
    page_url = moreview.get("href")
    print(page_url)
    page_urls.append(page_url)
# 크롤링에 사용한 브라우저를 종료합니다.
driver.close()


#239p
#상세보기 > 리뷰 정보 크롤링
            #별점,      리뷰
columns = ['score', 'review']
df = pd.DataFrame(columns=columns)
driver = webdriver.Chrome(ChromeDriverManager().install())
for page_url in page_urls:    
    # 상세보기 페이지에 접속합니다
    driver.get(page_url)
    time.sleep(2)    
    # 첫 페이지 리뷰를 크롤링합니다
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # contents_div = soup.find(name="div", attrs={"class":"evaluation_review"})  > 교재
    contents_div = soup.find(name="div", attrs={"class":"cont_evaluation"})    
    # 별점을 가져옵니다.
    # rates = contents_div.find_all(name="em", attrs={"class":"num_rate"})   > 교재
    rates = contents_div.find_all(name="span", attrs={"class":"txt_desc"})
    # 리뷰를 가져옵니다.
    reviews = contents_div.find_all(name="p", attrs={"class":"txt_comment"})
    
    for rate, review in zip(rates, reviews):
        row = [rate.text[0], review.find(name="span").text]
        series = pd.Series(row, index=df.columns)
        df = df.append(series, ignore_index=True)
    
    # 2-5페이지의 리뷰를 크롤링합니다
    for button_num in range(2, 6):
        # 오류가 나는 경우(리뷰 페이지가 없는 경우), 수행하지 않습니다.
        try:
            # another_reviews = driver.find_element_by_xpath("//a[@data-page='" + str(button_num) + "']") > 교재
            another_reviews = driver.find_element(By.XPATH, '//a[@class="link_more"]')
            another_reviews.click()
            time.sleep(2)           
            # 페이지 리뷰를 크롤링합니다
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            # contents_div = soup.find(name="div", attrs={"class":"evaluation_review"})  > 교재
            contents_div = soup.find(name="div", attrs={"class":"cont_evaluation"})   
            # 별점을 가져옵니다.
            # rates = contents_div.find_all(name="em", attrs={"class":"num_rate"})   > 교재
            rates = contents_div.find_all(name="span", attrs={"class":"txt_desc"})
            # 리뷰를 가져옵니다.
            reviews = contents_div.find_all(name="p", attrs={"class":"txt_comment"})

            for rate, review in zip(rates, reviews):
                row = [rate.text[0], review.find(name="span").text]
                series = pd.Series(row, index=df.columns)
                df = df.append(series, ignore_index=True)
        except:
            break    
driver.close()
# 4점 이상의 리뷰는 긍정 리뷰, 3점 이하의 리뷰는 부정 리뷰로 평가합니다.
#긍정리뷰 = 1, 부정리뷰 = 0
df['y'] = df['score'].apply(lambda x: 1 if float(x) > 3 else 0)
print(df.shape)
print(df.head())
df.to_csv("review_gangnam.csv", index=False)


#한글만
# 텍스트 정제 함수 : 한글 이외의 문자는 전부 제거
def text_cleaning(text):
    # 한글의 정규표현식으로 한글만 추출합니다.
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', str(text))
    return result
# 함수를 적용하여 리뷰에서 한글만 추출합니다.
df = pd.read_csv("review_gangnam.csv")
df['ko_text'] = df['review'].apply(lambda x: text_cleaning(x))
del df['review']
# 한 글자 이상의 텍스트를 가지고 있는 데이터만 추출합니다
df = df[df['ko_text'].str.len() > 0]
print("---한글만 추출---")
print(df.head())


#형태소단위로 추출
from konlpy.tag import Okt
# konlpy라이브러리로 텍스트 데이터에서 형태소를 추출합니다.
def get_pos(x):
    tagger = Okt()
    pos = tagger.pos(x)
    pos = ['{}/{}'.format(word,tag) for word, tag in pos]
    return pos
# 형태소 추출 동작을 테스트합니다.
result = get_pos(df['ko_text'].values[0])
print("---한글 + 형태소 단위 추출---")
print(result)


#분류모델 학습 데이터로 변환
from sklearn.feature_extraction.text import CountVectorizer
# 형태소를 벡터 형태의 학습 데이터셋(X 데이터)으로 변환합니다.
index_vectorizer = CountVectorizer(tokenizer = lambda x: get_pos(x))
X = index_vectorizer.fit_transform(df['ko_text'].tolist())
print(X.shape)
print("---학습 데이터 변환---")
print(str(index_vectorizer.vocabulary_)[:100]+"..")
print("---학습 데이터 변환---")
print(df['ko_text'].values[0])
print(X[0])


#변환 245p
from sklearn.feature_extraction.text import TfidfTransformer
# TF-IDF 방법으로, 형태소를 벡터 형태의 학습 데이터셋(X 데이터)으로 변환합니다.
tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)
print("---TF-IDF---")
print(X.shape)
print(X[0])


#긍정, 부정 리뷰 분류하기
#분류 모델링 : 데이터셋 분리
from sklearn.model_selection import train_test_split
y = df['y']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("---분류 모델링 : 데이터셋 분리---")
print(x_train.shape)
print(x_test.shape)

#246p
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 로지스틱 회귀모델을 학습합니다.
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred_probability = lr.predict_proba(x_test)[:,1]
# 로지스틱 회귀모델의 성능을 평가합니다.
print("---로지스틱 회귀모델 평가---")
print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision : %.3f" % precision_score(y_test, y_pred))
print("Recall : %.3f" % recall_score(y_test, y_pred))
print("F1 : %.3f" % f1_score(y_test, y_pred))

# Confusion Matrix를 출력합니다.
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("---Confusion Matrix---")
print(confmat)

# AUC를 계산합니다.
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_probability)
roc_auc = roc_auc_score(y_test, y_pred_probability)
print("AUC : %.3f" % roc_auc)

# ROC curve 그래프를 출력합니다.
plt.rcParams['figure.figsize'] = [5, 4]
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.3f)' % roc_auc, 
         color='red', linewidth=4.0)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Logistic regression')
plt.legend(loc="lower right")
plt.show()

# # 클래스 불균형이면 주석 풀고 진행
# #247p
# #클래스 불균형 문제 해결하기
# # 0, 1 얼마나 있는지 확인해보자.
# df['y'].value_counts()
# print("---0,1 얼마나 있는지 확인하기---")
# print(df['y'].value_counts())
# #248p
# #1:1
# positive_random_idx = df[df['y']==1].sample(50, random_state=10).index.tolist()
# negative_random_idx = df[df['y']==0].sample(50, random_state=10).index.tolist()
# #랜덤 데이터로 데이터셋 나눕니다
# random_idx = positive_random_idx + negative_random_idx
# sample_X = X[random_idx, :]
# y = df['y'][random_idx]
# x_train, x_test, y_train, y_test = train_test_split(sample_X,y,test_size=0.30)
# print("---클래스 불균형 문제 해결---")
# print(x_train.shape)
# print(x_test.shape)
# #248p
# #로지스틱 회귀 모델 다시 학습하기
# lr = LogisticRegression(random_state=0)
# lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
# y_pred_probability = lr.predict_proba(x_test)[:,1]
# # 로지스틱 회귀모델의 성능을 평가합니다.
# print("---로지스틱 회귀모델 평가---")
# print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
# print("Precision : %.3f" % precision_score(y_test, y_pred))
# print("Recall : %.3f" % recall_score(y_test, y_pred))
# print("F1 : %.3f" % f1_score(y_test, y_pred))
# #249p
# # Confusion Matrix를 출력합니다.
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print("---Confusion Matrix---")
# print(confmat)
# # //AUC 계산
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_probability)
# roc_auc = roc_auc_score(y_test, y_pred_probability)
# print("AUC : %.3f" % roc_auc)


##4 중요키워드 분석
#회귀 모델의 피처 영향력 추출
# 학습한 회귀 모델의 계수를 출력합니다.
plt.rcParams['figure.figsize'] = [10, 8]
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
plt.show()
print("---회귀 모델 계수---")
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[-5:])


# 중요 피처의 형태소
# 회귀 모델의 계수를 높은 순으로 정렬합니다. 
coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)
# 회귀 모델의 계수를 index_vectorizer에 맵핑하여, 어떤 형태소인지 출력할 수 있게 합니다.
invert_index_vectorizer = {v: k for k, v in index_vectorizer.vocabulary_.items()}
# 계수가 높은 순으로, 피처에 형태소를 맵핑한 결과를 출력합니다. 계수가 높은 피처는 리뷰에 긍정적인 영향을 주는 형태소라고 할 수 있습니다.
print("---계수 높은 순_형태소 출력---")
print(str(invert_index_vectorizer)[:100]+'..')


#251p
# 상위 20개 긍정 형태소를 출력합니다.
print("---top20 : 긍정 리뷰---")
for coef in coef_pos_index[:20]:
    print(invert_index_vectorizer[coef[1]], coef[0])

# 상위 20개 부정 형태소를 출력합니다.
print("---top20 : 부정 리뷰---")
for coef in coef_pos_index[-20:]:
    print(invert_index_vectorizer[coef[1]], coef[0])