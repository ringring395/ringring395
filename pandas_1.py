import pandas as pd
#판다스 사용 선언(별칭 : pd)
names = ["king", "kong", "sing", "ding"]
births = [123, 456, 111, 511]
custom = [10, 5, 22, 221115]
            # zip : 길이가 같은 데이터를 묶어주는 함수
DataSet = list(zip(names, births))
                                # columns : 컬럼명 지정
ds = pd.DataFrame(data = DataSet, columns=["NAMEs", "BIRTHs"])
print(ds.head())    #데이터 상단 출력
print(ds.dtypes)    #열타입 정보
print("데이터 인덱스 %s" %ds.index)     #인덱스 정보
print("데이터 열형태 %s" %ds.columns)   #열형태 정보
print("열 선택 %s" %ds["NAMEs"])  #열 선택
print("인덱스 선택 %s" %ds[0:2])      #인덱스 선택
#Births 열이 200보다 큰 데이터를 선택
print(ds[ds["BIRTHs"]>200])
print("평균값 계산 %s" %ds.mean())        #평균값 계산