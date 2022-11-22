# 길이가 3인 배열(python언어를 제외한 나머지 프로그래밍 언어)
# 길이가 3인 리스트(python언어에서 사용되는 명칭)
a = [1, 2, 3]
print(a[0])     # 1
print(a[-1])    # 3 : 뒤에서부터

#2차원 리스트
b = [1, 2, 3, ["a", "b", "c"]]
print(b[0])     # 1
print(b[3])     # ["a", "b", "c"]
print(b[-1])    # ["a", "b", "c"]
print(b[3][0])  # a
print(b[-1][0]) # a
print(b[3][-1]) # c

#3중 리스트
c = [11, 22, ["to", "day", ["Ding", "dong"]]]
print(c[2][2][0])   #Ding
print(c[-1])        #["to", "day", ["Ding", "dong"]]

#리스트 슬라이싱
a = [20,22,10,26,"wed"]
print(a[:2])    #[20,22]
print(a[2:])    #[10,26,"wed"]

#슬라이싱_2차원리스트
d = [11, 22, ["to", "day", "is"], 44, 55]
print(d[2:4])   #[["to", "day", "is"], 44]
print(d[2][:2]) #["to", "day"]

#연산하기
#연결 +
a = ["a", "b", "c"]
b = ["Ding", "dong"]
print(a+b)  #["a", "b", "c", "Ding", "dong"]

#반복 *
b = ["Ding", "dong"]
print(b*3)  #["Ding", "dong", "Ding", "dong", "Ding", "dong"]

#길이 구하기
print(len(a))
print(len(b))

#형변환
a = [11, 22, 33, 44, 55]
#print(a[2] + "scene") --> 오류 : int + str
print(str(a[2]) + "scene") #33scene

#리스트 값 수정 및 삭제
a = [20, 22, 10, 27]
#수정 : 리스트에서 1번째 값을 20 -> 2022
a[0] = 2022       
print(a)        #[2022, 22, 10, 27]
#삭제 : 리스트에서 2번째 값 
del a[1]
print(a)        #[2022, 10, 27]
#삭제
b = [11, 22, 33, 44, 55, 66, 77]
del b[3:]
print(b)        #[11, 22, 33]

#리스트 함수
a = [2022, 10, 27]
a.append("목")
print(a)        #[2022, 10, 27, "목"]
a.append([11, 22])
print(a)        #[2022, 10, 27, "목", [11, 22]]

#정렬
a = [500, 1000, 5000, 100, 10000, 50000]
a.sort()        #오름차순(기본값)
print(a)        #[100, 500, 1000, 5000, 10000, 50000]
a.sort(reverse=False)   #오름차순
print(a)        #[100, 500, 1000, 5000, 10000, 50000]
a.sort(reverse=True)    #내림차순
print(a)        #[50000, 10000, 5000, 1000, 500, 100]
#뒤집기
b = ["king", "ring", "sing", "ding"]
b.reverse()     
print(b)        #['ding', 'sing', 'ring', 'king']
#index
a = [500, 1000, 5000, 100, 10000, 50000]
print(a.index(10000))   # 인덱스번호 : 4
#print(a.index(3000))    # 없을 경우 : 에러
#insert
a = [2022, 10, 27]
a.insert(0, "Today")
print(a)        #['Today', 2022, 10, 27]
a.insert(1, "목")
print(a)        #['Today', '목', 2022, 10, 27]

#삭제
a = [2022, 10, 26, 10, 27, 10, 28, 10, 29]
a.remove(10)    #10삭제
print(a)        #[2022, 26, 10, 27, 10, 28, 10, 29]
a.remove(10)    #10삭제
print(a)        #[2022, 26, 27, 10, 28, 10, 29]

#pop
a = [2022, 10, 26, 10, 27, 10, 28, 10, 29]
a.pop()     #생략하면 마지막 인덱스 삭제됨
print(a)    #[2022, 10, 26, 10, 27, 10, 28, 10]
print(a.pop(1))    #인덱스 1번 꺼내고 삭제
print(a)    #[2022, 26, 10, 27, 10, 28, 10]

#count
s = "Smooth like butter"
print(len(s))       #s 문자열의 길이            -- 18
print(s.count("t")) #s 문자열에서 "t"의 갯수    -- 3
a = [2022, 10, 26, 10, 27, 10, 28, 10, 29]
print(len(a))       #a 리스트의 길이            -- 9
print(a.count(10))  #a 리스트에서 10의 갯수     -- 4

#리스트 확장
a = [2022, 10, 27]
a.extend([10, 28])
print(a)    #[2022, 10, 27, 10, 28]
a.extend([10, 29])
print(a)    #[2022, 10, 27, 10, 28, 10, 29]