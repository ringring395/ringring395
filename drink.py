import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# 데이터 경로
file_path = "python-data-analysis-master/data/drinks.csv"
# read_csv() 함수를 이용해 drinks.csv를 데이터프레임형태(엑셀)로 불러오기
drink = pd.read_csv(file_path)     # sep 기본값 = , 콤마
# print(drink.info())
# print(drink.head(10))

# #맥주 & 와인 [단순] 상관 분석
# #pearson : 상관계수를 구하는 방법 중 하나
# corr = drink[["beer_servings", "wine_servings"]].corr(method="pearson")
# print(corr)
# #[다중] 상관 분석
# cols = ["beer_servings", "spirit_servings", "wine_servings", "total_litres_of_pure_alcohol"]
# corr = drink[cols].corr(method="pearson")
# print(corr)
#히트맵 시각화
# cols_view = ["bear", "spirit", "wine", "alcohol"]
# sns.set(font_scale=1.5)
# #히트맵 설정
# hm = sns.heatmap(corr.values,       
#                 cbar = True,        #범위 색깔표
#                 annot = True,       #네모 안에 숫자 표시
#                 square = True,      #정사각형으로 출력
#                 fmt = ".2f",      #소수점 둘째자리까지
#                 annot_kws = {"size":15},
#                 yticklabels = cols_view,
#                 xticklabels = cols_view)
# plt.tight_layout()
# plt.show()
#산점도 그래프
# sns.set(style = "whitegrid", context = "notebook")
# sns.pairplot(drink[["beer_servings", "spirit_servings",
#             "wine_servings", "total_litres_of_pure_alcohol"]],
#             height = 2.5)          
# plt.show()
#72p
#전처리
# print("---isnull---")
# print(drink["continent"].isnull()[:10])  #결측 데이터 : True
# print("---fillna---")
#fillna : null값을 찾아서 대체문자를 삽입함
drink["continent"] = drink["continent"].fillna("OT")
# print(drink["continent"][:10])
#파이차트
#index :  대륙이름를 리스트화
#['AF', 'EU', 'AS', 'OT', 'OC', 'SA']
# labels = drink["continent"].value_counts().index.tolist()
# #values :  대륙별 국가수를 리스트화
# #[53, 45, 44, 23, 16, 12]
# fracs1 = drink["continent"].value_counts().values.tolist()
#             #4번째 값은 간격 벌림
# explode = (0, 0, 0, 0.25, 0, 0)
# #파이차트 정의
# plt.pie(fracs1, 
#         explode = explode,
#         labels = labels,
#         autopct = "%0.f%%",     #값 출력 형식 : 소수점없고 % 붙임
#         shadow = True)
# plt.title("null data to \'OT\'")
# plt.show()

#221118
#대륙별 spirit_servings의 통계적 정보는??
        #대륙별 정렬                           (평균, 최솟값, 최댓값, 합계) 계산
# ss_info = drink.groupby("continent").spirit_servings.agg(["mean", "min", "max", "sum"])
# print(ss_info.head())
# #현재 평균보다 많은 알코올을 섭취하는 대륙은??
# #total_mean : 총 알코올 소비량의 평균 = 현재 평균
# total_mean = drink.total_litres_of_pure_alcohol.mean()
# print("---현재 평균---")
# print(total_mean)
# #continetn_mean : 대륙별 총 알코올 소비량의 평균
# continent_mean = drink.groupby("continent")["total_litres_of_pure_alcohol"].mean()
# print("---대륙별 평균---")
# print(continent_mean)
# # 대륙별 평균 >= 현재평균
# continent_over_mean = continent_mean[continent_mean >= total_mean]
# print("---대륙별평균 >= 현재평균---")
# print(continent_over_mean)
# #평균 beer_servings가 가장 높은 대륙은??
#                                 #대륙별         맥주소비량   평균
# beer_continent = drink.groupby("continent").beer_servings.mean()
# print("---대륙별 맥주소비량 평균---")
# print(beer_continent)
#              #대륙별 맥주소비량평균   최댓값
# beer_continent_max = beer_continent.idxmax()
# print("---맥주소비량이 가장 높은 대륙---")
# print(beer_continent_max)

# #시각화 : 대륙별 spirit_servings의 통계적 정보(ss_info)
# n_group = len(ss_info.index)       #길이 6
# ss_mean = ss_info["mean"].tolist() #평균만
# ss_min = ss_info["min"].tolist() #최솟값
# ss_max = ss_info["max"].tolist() #최댓값
# ss_sum = ss_info["sum"].tolist() #합계
# #arange : 괄호만큼 반복한다
# index = np.arange(n_group)
# bar_width = 0.1 #세로막대의 굵기
# #세로막대 정의
# # rects1 = plt.bar(index, ss_mean, bar_width, color="r", label="Mean")
# # rects2 = plt.bar(index, ss_min, bar_width, color="g", label="Min")
# # rects3 = plt.bar(index, ss_max, bar_width, color="b", label="Max")
# # rects4 = plt.bar(index, ss_sum, bar_width, color="y", label="sum")
# #세로막대 정의2
#         #겹치지않게 위치를 조정해줌
# rects1 = plt.bar(index, ss_mean, bar_width, color="r", label="Mean")
# rects2 = plt.bar(index+bar_width, ss_min, bar_width, color="g", label="Min")
# rects3 = plt.bar(index+bar_width*2, ss_max, bar_width, color="b", label="Max")
# rects4 = plt.bar(index+bar_width*3, ss_sum, bar_width, color="y", label="sum")
# #막대그래프 정의
# plt.xticks(index, ss_info.index.tolist())
# plt.legend()
# plt.show()


# #시각화 : 현재 평균보다 많은 알코올을 섭취하는 대륙(continent_mean)
# continents = continent_mean.index.tolist()
# #리스트에 평균mean 추가
# continents.append("mean")
# #x축은 continents의 길이만큼 반복
# x_pos = np.arange(len(continents))
# alcohol = continent_mean.tolist()       #y축
# #y축에 전체 평균값total_mean 추가
# alcohol.append(total_mean)
# #막대그래프 정의
#                    #x축    y축        정렬            투명도
# bar_list = plt.bar(x_pos, alcohol, align="center", alpha=0.5)
# #전체 평균 막대 스타일링
# bar_list[len(continents) - 1].set_color("r")
# #점선그래프 정의                            검정 점선
# plt.plot([0., 6], [total_mean, total_mean], "k--")
# plt.xticks(x_pos, continents)
# plt.ylabel("total_litres_of_pure_alcohol")    
# plt.title("total_litres_of_pure_alcohol by Continent")
# plt.show()

# #시각화 :대륙간의 beer_servings
# beer_group = drink.groupby("continent")["beer_servings"].sum()
# continents = beer_group.index.tolist()  #x축
# y_pos = np.arange(len(continents))      
# alcohol = beer_group.tolist()           #y축
# #막대그래프 정의
# bar_list = plt.bar(y_pos, alcohol, align="center", alpha=0.5)
# bar_list[continents.index("EU")].set_color("r")
# plt.xticks(y_pos, continents)
# plt.ylabel("beer_servings")
# plt.title("beer_servings by Continent")
# plt.show()

#80p
#아프리카와 유럽 간의 맥주 소비량 차이를 검정
#아프리카
africa = drink.loc[drink["continent"] == "AF"]
#유럽
europe = drink.loc[drink["continent"] == "EU"]
tTestResult = stats.ttest_ind(africa["beer_servings"], 
                        europe["beer_servings"])
tTestResultDiffVar = stats.ttest_ind(africa["beer_servings"], 
                        europe["beer_servings"],
                        equal_var = False)
print("The t-statistic and p_value assuming equal variances is %.3f and %.3f." %tTestResult)
print("The t-statistic and p_value not assuming equal variances is %.3f and %.3f." %tTestResultDiffVar)

#대한민국은 얼마나 술을 독하게 마시는지 : 탐색코드
#total_servings :총소비량 = 맥주 + spirit + 와인소비량
drink["total_servings"] = drink["beer_servings"]+drink["spirit_servings"]+drink["wine_servings"]
print("---총소비량---")
print(drink["total_servings"])
#alcohol_rate : 알코올 비율 = 알코올 소비량 / 총 소비량
drink["alcohol_rate"] = drink["total_litres_of_pure_alcohol"] / drink["total_servings"]
print("---알코올 비율---")
print(drink["alcohol_rate"])
#알코올이 null이면 0으로 채우기
drink["alcohol_rate"] = drink["alcohol_rate"].fillna(0)
print("---알코올 비율 + null을 0으로 채움---")
print(drink["alcohol_rate"])
#알코올 비율을 기준으로 순위 매김
country_with_rank = drink[["country", "alcohol_rate"]]
print("---알코올 비율을 기준으로 순위 매김---")
print(country_with_rank)
#매긴 순위를 내림차순 정렬
country_with_rank = country_with_rank.sort_values(by=["alcohol_rate"], ascending=0)
print("---내림차순 정렬 top5출력---")
print(country_with_rank.head(15))

#국가별 순위 정보를 차트로 시각화                                       
country_list = country_with_rank.country.tolist()       #리스트화
rank = country_with_rank.alcohol_rate.tolist()          #리스트화
x_pos = np.arange(len(country_list))
#막대그래프 정의
bar_list = plt.bar(x_pos, rank)
                           #대한민국은 빨강색 표시
bar_list[country_list.index("South Korea")].set_color("r")
plt.ylabel("alcohol rate")
plt.title("liquor drink rank by country")
plt.axis([0, 200, 0, 0.3])      #x,y 축 표시되는 범위
#대한민국 주석 작업
korea_rank = country_list.index("South Korea")
#korea_alc_rate : 대한민국 알코올 비율
korea_alc_rate = country_with_rank[country_with_rank["country"] == "South Korea"]["alcohol_rate"].values[0]
#       주석
plt.annotate("South Korea : "+str(korea_rank+1),
        xy = (korea_rank, korea_alc_rate),              #화살표 방향
        xytext = (korea_rank+10, korea_alc_rate+0.05),  #문자,화살표 위치
        arrowprops = dict(facecolor="r", shrink=0.05))  #화살표 특성
                        #shrink : 텍스트로부터 어느 위치에서 시작하는가 (0~1)
plt.show()