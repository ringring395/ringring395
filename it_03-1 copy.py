import pandas as pd                 # csv 데이터를 엑셀형태로 데이터로 변환하기 위한 모듈
import numpy as np                  # pandas 모듈의 숫자데이터로 합계, 평균 등 연산을 하기 위한 모듈
import matplotlib.pyplot as plt     # 데이터로 차트만들기 위한 모듈

picher_file_path = 'python-data-analysis-master/data/picher_stats_2017.csv'
batter_file_path = 'python-data-analysis-master/data/batter_stats_2017.csv'
picher = pd.read_csv(picher_file_path)
batter = pd.read_csv(batter_file_path)

plt.hist(picher['연봉(2018)'], bins=100)        # 2018년 연봉 분포를 출력
# plt.show()

plt.boxplot(picher['연봉(2018)'])               # 2018년 연봉 분포를 출력
# plt.show()


# 희귀 분석에 사용할 피처 살펴보기
picher_features_df = picher[['승', '패', '세', '홀드', '블론', '경기', 
                            '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9',
                            'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 
                            'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']]

# 피처 각각에 대한 히스토그램을 출력
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20, 16]
    fig = plt.figure(1)

    # df의 column 갯수 만큼의 subplot을 출력
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()

# plot_hist_each_column(picher_features_df)


# 판다스 형태로 정의된 데이터를 출력할 때 scientific-notation이 아닌 float 모양으로 출력되게 해줌
pd.options.mode.chained_assignment = None

# 피처 각각에 대한 스켕일링을 수행하는 함수를 정의
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df

# 피처 각각에 대한 스케일링을 수행
scale_columns = ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', 
                 '볼넷/9', '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR',
                 '연봉(2017)']
picher_df = standard_scaling(picher, scale_columns)
picher_df = picher_df.rename(columns={'연봉(2018)': 'y'})
print(picher_df.head(5))


# 피처들의 단위 맞춰주기: 원-핫 인코딩
# 팀명 피처를 원-핫 인코딩으로 변환
team_encoding = pd.get_dummies(picher_df['팀명'])
picher_df = picher_df.drop('팀명', axis=1)
picher_df = picher_df.join(team_encoding)
print(team_encoding.head(5))



# 희귀 분석을 위한 학습, 테스트 데어터셋 분리
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# 학습 데이터와 테스트 데이터로 분리
X = picher_df[picher_df.columns.difference(['선수명', 'y'])]
y = picher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)



# 희귀 분석 계수학습 & 학습된 계수 출력
# 희귀 분석 계수를 학습(희귀 모델 학습)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 학습된 계수를 출력
print(lr.coef_)

# 어떤 피처가 가장 영향력이 강한 피처일까
import statsmodels.api as sm

# statsmodels 라이브러리로 희귀 분석을 수행.
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())



# 어떤 피처가 가장 영향력이 강한 피처일까
# 한글 출력을 위한 사전 설정 단계
import matplotlib as mpl
mpl.rc('font', family='NanumGothicOTF')
plt.rcParams['figure.figsize'] = [20, 16]

# 희귀 계수를 리스트로 변환
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

# 변수명을 리스트로 반환
x_labels = model.params.index.tolist()

# 희귀 계수를 출력
ax = coefs_series.plot(kind='bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)

plt.show()



# 예측 모델의 평가하기: R2 score
# 학습 데이터와 테스트 데이터로 분리
x = picher_df[picher_df.columns.difference(['선수명', 'y'])]
y = picher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# 희귀 분석 모델을 학습
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 희귀 분석 모델을 평가
print(model.score(X_train, y_train))    # train R2 score를 출력
print(model.score(X_test, y_test))      # test R2 score를 출력



# 예측 모델의 평가하기: RMSE score
# 희귀 분석 모델을 평가
y_predictions = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_predictions)))     # train RMSE score를 출력
y_predictions = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_predictions)))      # train RMSE score를 출력



# 피처들의 상관 관계 분석하기
import seaborn as sns

# 피처 간의 상관계수 행렬을 계산
corr = picher_df[scale_columns].corr(method='pearson')
show_cols = ['win', 'lose', 'save', 'hold', 'blon', 'match', 'start',
             'inning', 'strike3', 'ball4', 'homerun', 'BABIP', 'LOB',
             'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '2017']

# corr 행렬 히트맵을 시각화
plt.rc('font', family='NanumGothicOTF')
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values,
                 cbar = True,
                 annot = True,
                 square = True,
                 fmt = '.2f',
                 annot_kws = {'size': 15},
                 yticklabels = show_cols,
                 xticklabels = show_cols)

plt.tight_layout()
plt.show()



# 희귀 분석 예측 성능을 높이기 위한 방법: 다중 공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 피처마다의 VIF 계소를 출력
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))