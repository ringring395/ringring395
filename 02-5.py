#딕셔너리 c의 key로 1, value로 'big'을 저장해보자.
c = {1: "big"}
print(c)
c1 = {3:[1,2,3]}
print(c1)

# dic 추가하기
a = {1:"king"}
print(a)
a[2] = "kong"
print(a)
a["name"] = "king kong"
print(a)

# dic 저장된 값 사용하기
b = {"king":"kong", "sing":"song", "ding":"dong"}
print(b["king"])    #kong
print(b["sing"])    #song
print(b["ding"])    #dong

# key 중복일때
b = {"king":"kong", "king":"song", "ding":"dong"}
print(b)

# key 리스트 넣을때
b = {("king", "kong"):11}
print(b)
#b = {["king", "kong"]:11}
#print(b)

# key/value -> 리스트로 만들기
b = {"king":"kong", "sing":"song", "ding":"dong"}
b.keys()            #딕셔너리 b에서 key만 리스트로 따로 저장
print(b.keys())
print(b.values())   #딕셔너리 b에서 value만 리스트로 따로 저장
print(b.items())    #딕셔너리 b에서 (key, value) 리스트로 저장

# for k in b.keys():
#     print(k)
#     print("~~")
# print("end")    

# for k in b.values():
#     print(k)
#     print("~~")    
# print("byebye")    

#clear 함수
print(b.clear())    #딕셔너리 b에 있는 key, value 초기화
print(b)

#get 함수
b = {"king":"kong", "sing":"song", "ding":"dong"}
print(b["king"])        #king의 value값 호출
print(b.get("king"))    #king의 value값 호출 (get 함수 사용)
#print(b[1])            #에러
print(b.get(1))         #None (딕셔너리에 해당 key가 없을 때)
print(b.get(1,"a"))     #a (딕셔너리에 해당 key가 없으면 default값 출력됨)

#in 함수
print("sing" in b)      #True
print("ring" in b)      #False
print("song" in b)      #False