#집합 자료형
s1 = set([1,2,3,1,2])   #set 함수 list 자료형을 입력
print(s1)
s2 = set("Smooth")   #set 함수 문자열 자료형을 입력
print(s2)

ls = list(s1)   #set 자료형 ->list자료형 변환 후 ls에 저장
print(ls)
print(ls[0])
print(ls[1])
print(ls[2])

ts = tuple(s1) #set자료형 -> tuple자료형 변환 후 ts에 저장
print(ts)
print(ts[0])
print(ts[1])
print(ts[2])

s1 = set([1,2,3,4,5,6])
s2 = set([4,5,6,7,8,9])
#교집합 (순서 상관 X)
print(s1 & s2)
print(s1.intersection(s2))  # 교집합 함수 
print(s2 & s1)
print(s2.intersection(s1))  # 교집합 함수
#합집합 (순서 상관 X)
print(s1 | s2)
print(s1.union(s2))         # 합집합 함수
print(s2 | s1)
print(s2.union(s1))         # 합집합 함수
#차집합 (순서 상관 O, 결과 달라짐)
print(s1 - s2)
print(s1.difference(s2))    # 교집합 함수
print(s2 - s1)
print(s2.difference(s1))    # 교집합 함수

#기존에 있는 집합에 값 추가하기
s1 = set([1,2,3])
print(s1)
#s1.add(4)
print(s1)

#여러개의 값 추가
s1.update([4,5,6])
print(s1)

#특정 값 제거
s1.remove(2)
print(s1)