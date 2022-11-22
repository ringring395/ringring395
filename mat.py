import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#데이터
# names = ["king", "kong", "sing", "ding"]
# births = [123, 456, 111, 511]
# custom = [10, 5, 22, 221115]
# #판다스 작업
# DataSet = list(zip(names, births))
# ds = pd.DataFrame(data = DataSet, columns=["NAMEs", "BIRTHs"])
# x = ds["NAMEs"]
# y = ds["BIRTHs"]

# #막대그래프 출력
# plt.bar(x, y)     #막대그래프
# plt.xlabel("NAME")   #x축
# plt.ylabel("BIRTH")  #y축
# plt.title("Bar plot")    #그래프 제목
# plt.show()   #그래프 출력

np.random.seed(20021115)
#arange(시작점(생략=0), 끝점, 증감(생략=1))
x = np.arange(0.0, 100.0, 5.0)  #0~100, 5씩 증가
y = (x+1.5)+np.random.rand(20) * 50
#산점도 데이터 출력
plt.scatter(x, y, c="blue", alpha=1, label="scatter point")
plt.xlabel("x")   #x축 제목
plt.ylabel("y")   #y축 제목
plt.legend(loc="right")    #범례 위치방향
plt.title("scatter plot")    #그래프 제목
plt.show()        #그래프 출력
