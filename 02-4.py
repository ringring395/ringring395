t1 = (2022, 10, 27, "목")
print(t1)

#인덱싱
print(t1[0])    #2022
print(t1[3])    #"목"

#슬라이싱
print(t1[1:])   #인덱스번호 1 ~ 끝까지

#연산하기
t2 = (10, 28, 10, 29)
print(t1+t2)    #t1에 t2 결합

#곱하기
print(t1*2)

#길이 구하기
print(len(t1))
print(len(t2))
