from copy import copy   #module
#변수
#변수명 = 값
a = 1
b = "butter"
c = [11, 22, 33]

print(id(c))

a = [1,2,3]
b = a       #a의 list 값을 b에 저장
b = a[:]    #a리스트의 처음부터 끝까지
a[1] = 4
print(a)
print(b)
print(id(a))
print(id(b))

a = [11,22,33]
b = copy(a)     #copy모듈 사용
a[1] = 55
print(a)
print(b)
print(id(a))
print(id(b))

#a = "king"
#b = "kong"
a,b = ("king", "kong")
print(a)
print(b)

(a,b) = "king", "kong"
print(a)
print(b)

a,b = "king", "kong"
print(a)
print(b)

a=b="king kong"
print(a)
print(b)