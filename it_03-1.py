
import pandas as pd             #엑셀로 변환하기 위한 모듈
import numpy as np              #숫자데이터로 합계, 평균등 연산을 위한 모듈
import matplotlib.pyplot as plt #차트 만들기 위한 모듈
#추가 148p
#pip install scikit-learn --pre
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
#추가 150p
import statsmodels.api as sm
#추가 152p
import matplotlib as mpl
#추가 155p
import seaborn as sns
#추가 157p
from statsmodels.stats.outliers_influence import variance_inflation_factor

#데이터 심기
picher_file_path = "python-data-analysis-master/data/picher_stats_2017.csv"
batter_file_path = "python-data-analysis-master/data/batter_stats_2017.csv"
# read_csv() 함수를 이용해 drinks.csv를 데이터프레임형태(엑셀)로 불러오기
picher = pd.read_csv(picher_file_path)
batter = pd.read_csv(batter_file_path)
#테스트용으로 컬럼을 출력해보자
# print(picher.columns)       #picher_stats_2017.csv 컬럼 정보
# print(batter.columns)
# print(picher.head())        #행 단위 출력(생략: 상위 5행)
# print(picher.shape)         #행,열 수 출력(행, 수)
# print(picher['연봉(2018)'].describe())  #연봉2018 통계(건수, 평균, 표준표차,,,)
#막대그래프
# plt.hist(picher['연봉(2018)'], bins=100)   #연봉2018 분포 출력
# plt.show()
#상자그림
# plt.boxplot(picher["연봉(2018)"])       #연봉2018 분포 출력
# plt.show()

#전체 히스토그램
picher_features_df = picher[["승", "패","세","홀드","블론","경기",
                        "선발","이닝","삼진/9","볼넷/9","홈런/9",
                        "BABIP","LOB%","ERA","RA9-WAR","FIP",
                        "kFIP","WAR","연봉(2018)","연봉(2017)"]]
#피처 각각에 대한 히스토그램
def plot_hist_each_column(df) :
    plt.rcParams["figure.figsize"] = [20,16]
            #전체 틀
    fig = plt.figure(1)                     
    #열 개수 만큼 subplot 출력
    for i in range(len(df.columns)) :
                #작은 그래프 하나하나
        ax = fig.add_subplot(5, 5, i+1)     
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()
# plot_hist_each_column(picher_features_df)

#예측, 투수의 연봉 예측하기
#float타입으로 소수점까지 디테일하게 표현하자
pd.options.mode.chained_assignment = None
#피처 스케일링 함수 정의
def standard_scaling(df, scale_columns) :
    for col in scale_columns :
        series_mean = df[col].mean()        #평균
        series_std = df[col].std()          #표준편차
                                            #(x-x평균)/x표준편차
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df

scale_columns = ['승', '패', '세', '홀드', '블론', '경기', 
                '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9', 
                'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 
                'kFIP', 'WAR', '연봉(2017)']  
picher_df = standard_scaling(picher, scale_columns)
picher_df = picher_df.rename(columns={"연봉(2018)": "y"})
print(picher_df.head())

#one-hot-encoding
#팀명을 원핫인코딩 하기
team_encoding = pd.get_dummies(picher_df["팀명"])
picher_df = picher_df.drop("팀명", axis=1)
picher_df = picher_df.join(team_encoding)
print(team_encoding.head())

#학습 데이터와 테스트 데이터로 분리
X = picher_df[picher_df.columns.difference(["선수명", "y"])]
y = picher_df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
#회귀 분석 계수 학습
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
#학습된 계수 출력
print(lr.coef_)

#statsmodel 라이브러리로 회귀 분석을 수행
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# 한글 출력 사전 설정 단계
mpl.rc("font", family="NanumGothicOTF")
plt.rcParams["figure.figsize"] = [20, 16]
#회귀 계수를 리스트로 반환
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)
#변수명을 리스트로 반환
x_labels = model.params.index.tolist()
# 회귀 계수를 출력
ax = coefs_series.plot(kind="bar")
ax.set_title("feature_coef_graph")
ax.set_xlabel("x_features")
ax.set_ylabel("coef")
ax.set_xticklabels(x_labels)
#plt.show()

#회귀분석모델 평가
X = picher_df[picher_df.columns.difference(["선수명", "y"])]
y = picher_df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print("---학습데이터 점수---")
print(model.score(X_train, y_train))
print("---테스트 데이터 점수---")
print(model.score(X_test, y_test))

#회귀 분석 모델 평가 RSME score
y_predictions = lr.predict(X_train)
print("---train RMSE score---")
print(sqrt(mean_squared_error(y_train, y_predictions)))
y_predictions = lr.predict(X_test)
print("---test RMSE score---")
print(sqrt(mean_squared_error(y_test, y_predictions)))

#피처 간의 상관계수 행렬 계산
corr = picher_df[scale_columns].corr(method='pearson')
show_cols = ['win', 'lose', 'save', 'hold', 'blon', 'match', 'start',
             'inning', 'strike3', 'ball4', 'homerun', 'BABIP', 'LOB',
             'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '2017']
#corr 행렬 히트맵 시각화
plt.rc("font",family="NanumGothic")
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values,
                cbar = True,
                annot=True,
                square=True,
                fmt=".2f",
                annot_kws={"size":15},
                yticklabels=show_cols,
                xticklabels=show_cols)
plt.tight_layout()
plt.show()

#157p
#회귀 분석 예측 성능을 높이기 위한 방법_ 다중 공선성 확인
# 투수마다 VIF 계수 출력
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))