#112p
print("---112p 1번---")
#1. 홍길동씨의 과목별 점수는 각각 다음과 같다
#국어=80
#영어=75
#과학 =55
#홍길동씨의 평균점수
sum=80+75+55
print(sum/3)

print("---112p 2번---")
#2. 자연수 13이 홀수인지 짝수인지 판별해보자
#짝수면 0, 홀수면 1을 출력
a = 13
if a%2==0 :
    print(0)
else:
    print(1)

print("---112p 3번---")
#홍길동 주민번호를 연월일(yyyymmdd) 부분과 
# 그 뒤 숫자부분으로 나눠서 출력해보자.
pin = "021025-3123456"
yyyymmdd = pin[:6]
num = pin[7:]
print(yyyymmdd)
print(num)

print("---113p 4번---")
#주민번호 뒷자리 맨 첫번째는 성별인데, 성별만 출력해보자.
print(pin[7])
print(num[0])

print("---113p 5번---")
#a:b:c:d 문자열을 replace함수로 a#b#c#d로 바꿔라.
a = "a:b:c:d"
b = a.replace(":", "#")
print(b)

print("---113p 6번---")
#[1,3,5,4,2]리스트를 [5,4,3,2,1]로 만들어라.
a = [1,3,5,4,2]
a.sort()
a.sort(reverse=True)
print(a)
    
print("---114p 7번---")
# ["Life", "is", "too", "short"]리스트를 Life is too short 로 만들자.
a = ["Life", "is", "too", "short"]
result = " ".join(a)
print(result)

print("---114p 8번---")
#(1,2,3)에 4를 추가해서 (1,2,3,4)를 만들자.
a = (1, 2, 3)
a = a+(4,)
print(a)

print("---114p 9번---")
# a 딕셔너리에 관해 오류가 발생하는 경우는?
a = dict()
a['name'] = 'pyhon'
a[('a')] = 'python'
# a[[1]] = 'python' 리스트값은 안됨
a[250] = 'python'
print(a)

print("---115p 10번---")
# 딕셔너리 a에서 'B'에 해당되는 값은
a = {'A':90, 'B':80, 'C':70}
result = a.pop('B')
print(a)
print(result)

print("---115p 11번---")
# a 리스트에서 중복 숫자 제거
a = [1,1,1,2,2,3,3,3,4,4,5] #리스트형
aSet = set(a)
b = list(aSet)
print(b)

print("---115p 12번---")
# a,b선언 하고 a 두번째요소값 변경하면 b에선 어떻게 될까.
a = b = [1,2,3]
a[1] = 4
print(a)
print(b)

#146p
print("---146p 1번---")
#다음 코드 결과는?
a = "Life is too short, you need python"
if "wife" in a : 
    print("wife")
elif "python" in a and "you" not in a : 
    print("pyhon")
elif "shirt" not in a : 
    print("shirt")
elif "need" in a : 
    print("need")
else : 
    print("none")
#shirt 출력    

print("---146p 2번---")
# while문 사용 :  1~1000까지의 자연수 중 3배수 합은?
result = 0
i = 1
while i <=1000 :
    if i%3 == 0 :
        result += i
    i += 1
print(result)

print("---147p 3번---")
# while문 사용 : 별 표시
i = 0
while True :
    i += 1
    if i > 5 :
        break
    for j in range(1, i+1) :
        print("*", end="")
    print("")                   #줄바꿈

print("---147p 4번---")
# for문 사용 : 1~100까지 숫자 출력
for i in range(1, 101) :
    print(i)

print("---148p 5번---")
# for문 사용 : A 학급의 평균 점수는?
A = [70,60,55,75,95,90,80,80,85,100]
total = 0
for score in A :
    total += score
average = total/10
print(average)

print("---148p 6번---")
# 리스트 내포 사용 : 홀수에만 2곱하여 저장
numbers =[1,2,3,4,5]
result =[]
for n in numbers :
    if n % 2 == 1 :
        result.append(n*2)
print(result)
#리스트 내포
numbers=[1,2,3,4,5]
result = [n*2 for n in numbers if n%2==1]
print(result)

#179p
print("---179p 1번---")
# 홀수인지 자연수인지 판별
def is_odd(number) :
    if number%2 ==1 :   #2로 나눴을때 나머지가 1이면 홀수
        return True
    else :
        return False
print(is_odd(27))
print(is_odd(20))

print("---179p 2번---")
# 입력되는 수의 평균값 계산
def avg_numbers(*args) :
    result = 0
    for i in args :
        result += i
    return print(result/len(args))  #매개변수를 모두 더하고, 그 수만큼 나눠서 평균구함
avg_numbers(1,2)
avg_numbers(1,2,3,4,5)
avg_numbers(90,85,80)

print("---179p 3번---")
# 숫자 2개를 입력받고, 더해서 돌려주는 프로그램
# input1 = int(input("첫번째 숫자를 입력하세요: "))
# input2 = int(input("두번째 숫자를 입력하세요: "))
# total = input1 + input2
# print("두 수의 합은 %s 입니다." %total)

print("---180p 4번---")
#결과가 다른 하나는?
print("you" "need" "python")            #youneedpython
print("you"+"need"+"python")            #youneedpython
print("you", "need", "python")          #you need python
print("".join(["you", "need", "python"]))#youneedpython

print("---180p 5번---")
# test.txt 파일에 Life is too short 문자열 저장하고 
# 다시 그 파일 읽어서 출력하자.
# f1 = open("test.txt", "w")
# f1.write("Life is too short")
# f1.close()

# f2 = open("test.txt", "r")
# print(f2.read())
# f2.close()

print("---181p 6번---")
# 입력 내용을 test.txt에 추가해서 저장
# user_input = input("저장할 내용을 입력하세요: ")
# f = open("test.txt", "a")
# f.write(user_input)
# f.write("\n")           #줄단위 구분위해 줄바꿈 문자 삽입
# f.close()

print("---181p 7번---")
#test.txt 파일에서 java 문자열을 python으로 바꿔서 저장
# f = open("test.txt", "r")   #test.txt파일의 내용을 읽기모드로 열기
# body = f.read()             #전체 내용을 읽어서 body에 저장
# f.close()

# body = body.replace("java", "python")

# f = open("test.txt", "w")   # test.txt파일을 수정모드로 열기
# f.write(body)               # body를 입력하기
# f.close()

#262p
print("---262p 1번---")
# 상속하는 UpgradeCalculator 만들고,
# 값을 뺄수있는 minus 메소드 추가
class Calculator:
    #생성자
    def __init__(self) :
        self.value =0
    def add(self, val):
        self.value += val
    #자식클래스         (부모클래스)                       
class UpgradeCalculator(Calculator):
    def minus(self, val):
        self.value -= val

cal = UpgradeCalculator()
cal.add(10)     #0 + 10 = 10
cal.minus(7)    #10 - 7 = 3
print(cal.value)

print("---262p 2번---")
# value가 100이상 값은 가질 수 없도록 제한하는
# MaxLimitCalculator 클래스 만들자.
    #자식클래스          (부모클래스)  
class MaxLimitCalculator(Calculator):
    #부모클래스의 add함수를 발전시키자.
    def add(self, val) :
        self.value += val
        #self.value값이 100보다 크면
        if self.value >= 100 :
            #100으로 저장
            self.value = 100         
        return self.value

cal = MaxLimitCalculator()
cal.add(50) # 0 + 50 = 50
cal.add(60) #50 + 60 = 110
print(cal.value)    #100출력

print("---263p 3번---")
# 결과값 예측해보기
q3_1 = all([1,2,abs(-3)-3])
print(q3_1)   #0이 있으면 False
q3_2 = chr(ord('a')) == 'a'
print(q3_2)     #아스키로 바꿨다가(ord) 다시 문자열로 바꿈(chr)

print("---263p 4번---")
# filter, lambda 사용 : [1, -2, 3, -5, 8, -3]에서 음수 제거하자
def Q4(l) :
    return l > 0
print(list(filter(Q4, [1, -2, 3, -5, 8, -3])))
print(list(filter(lambda x:x>0, [1, -2, 3, -5, 8, -3])))

print("---263p 5번---")
# 234(10진수)의 16진수를 구하고,
# 0xea(16진수)의 10진수를 구하라
h = hex(234)
print(h)
i = int("0xea", 16)
print(i)

print("---264p 6번---")
# map, lambda 사용 : [1,2,3,4] 각 요소에 3 곱해진 리스트 만들자.
def Q6(x) :
    return x*3
print(list(map(Q6, [1,2,3,4])))
print(list(map(lambda x:x*3, [1,2,3,4])))

print("---264p 7번---")
# 최댓값, 최솟값 구하자
Q7 = [-8, 2, 7, 5, -3, 5, 0, 1]
print(max(Q7))
print(min(Q7))

print("---264p 8번---")
# 17/3의 결과를 소숫점 4자리까지 반올림해서 표시
print(17/3)
print(round(17/3, 4))

print("---264p 9번---")
#입력값 전부 더해서 출력하는 코드
# import sys
# numbers = sys.argv[1:]
# result =0
# for number in numbers :
#     result += int(number)
# print(result)    

print("---265p 10번---")
# os 모듈 사용해서 다음과 같이 동작하도록 코드 작성

print("---265p 11번---")
# glob모듈 사용 : .py 파일만 출력
import glob
print(glob.glob("D:/01-STUDY/pythonspace/*.txt"))

print("---265p 12번---")
# time 모듈 사용 : 현재날짜, 시간 출력
import time
print(time.strftime("%Y/%m/%d %H:%M:%S"))

print("---265p 13번---")
#random 모듈 : 로또번호 6개 생성.
import random
result =[]
while len(result) <6:
    num = random.randint(1, 45)
    if num not in result :
        result.append(num)
print(result)        
#221114
#322p
#1번
# q1 = "a:b:c:d"
# print("#".join(q1.split(":")))
# #2번
# q2 = {"A":90, "B":80}
# print(q2.get("c",70))
# #3번
# q3_1 = [1,2,3]
# q3_1 = q3_1 + [4,5]
# print(q3_1)
# q3_2 = [1,2,3]
# q3_2.extend([4,5])
# print(q3_2)
# #4번
# q4 = [20,55,67,82,45,33,90,87,100,25]
# result_q4 = 0
# while q4:
#     ok = q4.pop()
#     if ok >= 50 :
#         result_q4 += ok
# print(result_q4)
# #5번
# def fib(n) :
#     if n == 0 : return 0
#     if n == 1 : return 1
#     return fib(n-2) + fib(n-1)

# #6번
# q6 = input("숫자를 입력하세요 > ")
# num6 = q6.split(",")
# total = 0
# for n in num6 :
#     total += int(n)
# print(total)

# #7번
# def q7(n) :
#     result_q7=[]
#     i = 1
#     while i < 10 :
#         result_q7.append(n*i)
#         i = i+ 1
#     return result_q7   
#     #타입 변환 필요
# gu = int(input("한줄 구구단 숫자 입력하기 >"))
# print(q7(gu))
#8번

#9번

#10번