import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 데이터 경로
file_path = "python-data-analysis-master/data/chipotle.tsv"
# read_csv() 함수를 이용해 chipotle.tsv를 데이터프레임형태(엑셀)로 불러오기
chipo = pd.read_csv(file_path, sep="\t")    # sep : 구분기호

# print(chipo.shape)
# print("--------------------")
# print(chipo.info())

# print(chipo.head(10))       #처음부터 n개의 데이터 확인
# print(chipo.columns)        # 행 정보
# print("--------------------")
# print(chipo.index)          # 열 정보

# chipo["order_id"]=chipo["order_id"].astype(str)
# print(chipo.describe())

# print(len(chipo["order_id"].unique()))
# print(len(chipo["item_name"].unique()))
#주문량 top10 출력하기 
# item_count = chipo["item_name"].value_counts()[:10]
# for idx, (val, cnt) in enumerate(item_count.iteritems(), 1) :    
#     print("Top", idx, ":", val, cnt)
#아이템(item_name)별 주문개수(count)와 총량(quantity)
#주문개수
# order_count = chipo.groupby("item_name")["order_id"].count()[:10]
# print(order_count)     #10개씩 출력
#주문수량 합계
# item_quantity = chipo.groupby("item_name")["quantity"].sum()[:10]
# print(item_quantity) #10개씩 출력
# # 시각화
# item_name_list = item_quantity.index.tolist()
# print(item_name_list)
# x_pos = np.arange(len(item_name_list))  # x축
# y_pos = item_quantity.values.tolist()   # y축

# plt.bar(x_pos, y_pos, align="center")   #중앙정렬
# plt.ylabel("order_item_count")          #y축 제목
# plt.title("Distribution of all orderd item")    #그래프 제목
# plt.show()
#전처리 57p
print(chipo.info())
# quan_sum = chipo["quantity"].sum()  #주문수량의 합계
# print("-----주문수량의 합계-----")
# print(quan_sum)
# price_sum = chipo["item_price"].sum()  #주문 아이템 가격의 합계
# print("-----주문 아이템 가격의 합계-----")
# print(price_sum)
#전처리!!
# print("----전처리 before-----")
# print(chipo["item_price"])
# print("----전처리 after-----")
chipo["item_price"] = chipo["item_price"].apply(lambda x:float(x[1:]))
# print(chipo["item_price"])
# print(chipo.describe())     #숫자형 자료 통계
#탐색적 분석
#주문당 평균금액
                                #합계를 구하고 평균을 구하라
# print(chipo.groupby("order_id")["item_price"].sum().mean())
#10$이상 지불한 주문번호
chipo_orderid_group = chipo.groupby("order_id").sum()
price_10 = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print("----10불이상 지불한 주문번호-----")
print(price_10)
print("----10개만 출력-----")
print(price_10[:10])    #10개만 출력
print("----id만 10개 출력-----")
print(price_10[:10].index.values)
