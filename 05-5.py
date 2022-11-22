#내장함수
#abs
print(abs(3))       #3
print(abs(-3))      #3
print(abs(-3.1))    #3.1

#all
#       True && True && True => True
print(all([1,    2,     3])) 
#       True && True && True && False => False
print(all([1,    2,     3,      0])) 

#any
#       True || True || True => True
print(any([1,    2,     3])) 
#       True || True || True || False => True
print(any([1,    2,     3,      0])) 
#       False || False  => False
print(any([0,    ""])) 

#chr
print(chr(99))      #c
print(chr(104))     #h
print(chr(114))     #r

#dir
# print("리스트 %s" %dir([1,2,3]))
# print("튜플 %s" %dir((1,2,3)))
# print("딕셔너리 %s" %dir({1,2,3}))     #딕셔너리

#divmod
print(divmod(7,3))
print(7/3)      #나눗셈
print(7//3)     #몫
print(7%3)      #나머지

#enumerate
print(["king", "sing", "ding"])
for i, name in enumerate(["king", "sing", "ding"]) :
    print(i, name)

#eval
print(eval("1+2"))
print(eval("'king'+'kong'"))
print(eval("divmod(4,3)"))

#filter
def positive(l) :               #positive함수 선언
    result = []
    for i in l :
        if i > 0 :              #0보다 크면
            result.append(i)    #리스트에 추가됨
    return result
#positive함수 호출
print(positive([1, -10, 0, 50, 100]))   #1, 50, 100

#filter함수 사용
def positiveFilter(l) :
    return l > 0
#                   함수명,       함수에 차례대로 들어갈 자료형
print(list(filter(positiveFilter, [1, -10, 0, 50, 100])))

#hex
#10 -> 16진수
print(hex(1024))    #0x400
print(hex(8))      #0x8

#oct    10 -> 8진수
print(oct(1024))    #0o2000
print(oct(8))      #0o10

#id
a=3
print(id(3))
print(id(a))
b=a
print(id(b))

#input
# a = input("누구세요?? > ")
# print(a)

#int
print(int("3"))             #문자열 숫자
print(int(3.141592))        #소수점 숫자

print("3"+"4")              #문자열 연결
print(int("3")+int("4"))    #숫자 연산

print(int("11", 2))         #2진수 11 -> 10진수 변환
print(int("1A", 16))        #16진수 1A -> 10진수 변환

#isinstance
class Person :
    pass
class Animal :
    pass

#java : Person a = new Person()
a = Person()    #Person클래스를 인스턴스화

print(isinstance(a, Person))    #a가 Person의 인스턴스인가요? / True
print(isinstance(b, Person))    #b가 Person의 인스턴스인가요? / False
print(isinstance(a, Animal))    #a가 Animal의 인스턴스인가요? / False

#len
print(len("ringring"))              #8  
print(len("Smooth like butter"))    #18
print(len([1, -10, 0, 50, 100]))    #5
print(len((11, 22, 33)))            #3

#list
print(list("ringring"))
print(list((11, 22, 33)))

#tuple
print(tuple("ringring"))
print(tuple([11, 22, 33]))

#map(함수, list or tuple)
def two(numlist) :
    result = []
    for num in numlist :
        result.append(num*2)    #인수에 2를 곱해서 리스트에 추가
    return result

result = two([11, 22, 33, 44])
print(result)

#map 함수 사용
def twoMap(numlist) :
    return numlist*2
#위의 결괏값처럼 리스트로 출력
print(list(map(twoMap, [11, 22, 33, 44])))

#max
print(max([11, 22, 33]))
print(max(["k", "s", "d"]))
print(max("king"))

#min
print(min([11, 22, 33]))
print(min(["k", "s", "d"]))
print(min("king"))

#ord
print(ord("c"))      #99
print(ord("h"))     #104
print(ord("r"))     #114

#pow
print(pow(3,4))
print(pow(2,10))

#range
print(list(range(5)))   #0~4 1씩 증가
print(list(range(5, 10)))   #5~9 1씩 증가
print(list(range(1, 10, 2)))   #1~10 2씩 증가

#round
print(round(4.5))
print(round(3.141592))
print(round(3.141592, 3))

#sorted
print(sorted([3, 1, 2]))
print(sorted([22, 44, 55, 11]))

#str
print(str(3))

#sum
print(sum([1, 2, 3]))   #리스트
print(sum((1, 2, 3)))   #튜플

#type
print(type(1))
print(type((1, 2, 3)))
print(type(["k", "s", "d"]))
print(type("kingkong"))

#zip
print(list(zip([1,2,3], [11,22,33])))
print(list(zip(["king", "sing", "ding"], ("kong", "song", "dong"))))