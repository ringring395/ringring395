#카카오맵 감성분류 : 울산 삼산점 고기집
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

#카카오맵 크롤링
source_url = "https://map.kakao.com/"
driver = webdriver.Chrome(ChromeDriverManager().install())
#카카오 지도에 접속
driver.get(source_url)      
searchbox = driver.find_element(By.XPATH, "//input[@id='search.keyword.query']")
searchbox.send_keys("울산 삼산점 고기집")
# 검색버튼을 눌러서 결과를 가져옵니다
searchbutton = driver.find_element(By.XPATH,"//button[@id='search.keyword.submit']")
driver.execute_script("arguments[0].click();", searchbutton)
time.sleep(2)
# 검색 결과의 페이지 소스를 가져옵니다
html = driver.page_source
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
    contents_div = soup.find(name="div", attrs={"class":"cont_evaluation"})    
    # 별점을 가져옵니다.
    rates = contents_div.find_all(name="span", attrs={"class":"txt_desc"})
    # 리뷰를 가져옵니다.
    reviews = contents_div.find_all(name="p", attrs={"class":"txt_comment"})
    
    for rate, review in zip(rates, reviews):
        row = [rate.text[0], review.find(name="span").text]
        series = pd.Series(row, index=df.columns)
        df = df.append(series, ignore_index=True)

    for button_num in range(2, 6):
        # 오류가 나는 경우(리뷰 페이지가 없는 경우), 수행하지 않습니다.
        try:
            another_reviews = driver.find_element(By.XPATH, '//a[@class="link_more"]')
            another_reviews.click()
            time.sleep(2)           
            # 페이지 리뷰를 크롤링합니다
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            contents_div = soup.find(name="div", attrs={"class":"cont_evaluation"})   
            # 별점을 가져옵니다.
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
df.to_csv("review_sujin.csv", index=False)

#한글만
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', str(text))
    return result
df = pd.read_csv("review_sujin.csv")
df['ko_text'] = df['review'].apply(lambda x: text_cleaning(x))
del df['review']
# 한 글자 이상의 텍스트를 가지고 있는 데이터만 추출합니다
df = df[df['ko_text'].str.len() > 0]
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
print(result)

#분류모델 학습 데이터로 변환
from sklearn.feature_extraction.text import CountVectorizer
# 형태소를 벡터 형태의 학습 데이터셋(X 데이터)으로 변환합니다.
index_vectorizer = CountVectorizer(tokenizer = lambda x: get_pos(x))
X = index_vectorizer.fit_transform(df['ko_text'].tolist())
print(X.shape)
print(str(index_vectorizer.vocabulary_)[:100]+"..")
print(df['ko_text'].values[0])
print(X[0])

from sklearn.feature_extraction.text import TfidfTransformer
# TF-IDF 방법으로, 형태소를 벡터 형태의 학습 데이터셋(X 데이터)으로 변환합니다.
tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)
print(X.shape)
print(X[0])

#긍정, 부정 리뷰 분류하기
#분류 모델링 : 데이터셋 분리
from sklearn.model_selection import train_test_split
y = df['y']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(x_train.shape)
print(x_test.shape)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 로지스틱 회귀모델을 학습합니다.
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred_probability = lr.predict_proba(x_test)[:,1]
# 로지스틱 회귀모델의 성능을 평가합니다.
print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision : %.3f" % precision_score(y_test, y_pred))
print("Recall : %.3f" % recall_score(y_test, y_pred))
print("F1 : %.3f" % f1_score(y_test, y_pred))
# Confusion Matrix를 출력합니다.
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
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

#회귀 모델의 피처 영향력 추출
# 학습한 회귀 모델의 계수를 출력합니다.
plt.rcParams['figure.figsize'] = [10, 8]
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
plt.show()
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)[-5:])

# 중요 피처의 형태소
# 회귀 모델의 계수를 높은 순으로 정렬합니다. 
coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)
invert_index_vectorizer = {v: k for k, v in index_vectorizer.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'..')

# 상위 20개 긍정 형태소를 출력합니다.
for coef in coef_pos_index[:20]:
    print("긍정 : "+invert_index_vectorizer[coef[1]], coef[0])
# 상위 20개 부정 형태소를 출력합니다.
for coef in coef_pos_index[-20:]:
    print("부정 : "+invert_index_vectorizer[coef[1]], coef[0])