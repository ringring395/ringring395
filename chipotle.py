import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 데이터 경로
file_path = "python-data-analysis-master/data/chipotle.tsv"
# read_csv() 함수를 이용해 chipotle.tsv를 데이터프레임형태(엑셀)로 불러오기
chipo = pd.read_csv(file_path, sep="\t")    
#전처리
chipo["item_price"] = chipo["item_price"].apply(lambda x: float(x[1:]))
#//탐색적 분석
#10$이상 지불한 주문번호
# chipo_orderid_group = chipo.groupby("order_id").sum()
# price_10 = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
# print("----10불이상 지불한 주문번호-----")
# print(price_10)
# print("----10개만 출력-----")
# print(price_10[:10])    #10개만 출력
# print("----id만 10개 출력-----")
# print(price_10[:10].index.values)

#221117
#각 아이템 가격 구하기
#아이템 주문량이 1 구하기
# chipo_one_item = chipo[chipo.quantity==1]
# #그룹별 최저가 계산
# price_per_item = chipo_one_item.groupby("item_name").min()
# #가격을 기준으로 정렬
#                         #item_price 기준으로 (ascending=False)내림차순
# print(price_per_item.sort_values(by="item_price", ascending=False)[:10])

# #아이템 가격 분포 차트
# item_name_list = price_per_item.index.tolist()
# x_pos = np.arange(len(item_name_list))      #x축
# item_price = price_per_item["item_price"].tolist()  #y축
# plt.bar(x_pos, item_price, align="center")  #차트 정의
# plt.ylabel("item_price($)")                 #y축 제목
# plt.title("Distribution of item price")     #차트 제목
# plt.show()
# #가격 히스토그램 차트
# plt.hist(item_price)                     #차트 정의
# plt.ylabel("counts")                     #y축 제목
# plt.title("Histogram of item price")     #차트 제목
# plt.show()

#가장 비싼 주문에서 몇개의 아이템이 팔렸는가.
            #주문번호별 합계    ->     가격별 정렬     -> 내림차순
# print(chipo.groupby("order_id").sum().sort_values(by="item_price", ascending=False)[:10])

#Veggie Salad Bowl 주문횟수
#Veggie Salad Bowl이 몇번 주문되었는가
# chipo_salad = chipo[chipo["item_name"]=="Veggie Salad Bowl"]
# #한 주문에서 중복된 거 삭제
# chipo_salad = chipo_salad.drop_duplicates(["item_name", "order_id"])
# print(len(chipo_salad))     # 횟수 구하기
# print(chipo_salad.head(5))  # 5개만 출력해보기

#Chicken Bowl 2개 이상 주문한 고객의 Chicken Bowl 총 주문 수량
#item_name이 Chicken Bowl인 데이터
chipo_ch = chipo[chipo["item_name"]=="Chicken Bowl"]
#order_id 기준으로 수량 합계
chipo_ch_ordersum = chipo_ch.groupby("order_id").sum()["quantity"]
#chipo_chicken_ordersum이 2개이상인 것만
chipo_ch_result = chipo_ch_ordersum[chipo_ch_ordersum >= 2]
print(len(chipo_ch_result))        # 개수 구하기
print(chipo_ch_result.head(5))     # 5개만 출력