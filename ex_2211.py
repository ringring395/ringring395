#1번
# a = "Life is too short, you need python"

# if "wife" in a : print("wife")
# elif "python" in a and "you" not in a : print("pyhon")
# elif "shirt" not in a : print("shirt")
# elif "need" in a : print("need")
# else : print("none")

# #2번
# result = 0
# i = 1
# while i <=1000 :
#     if result%3 == 0 :
#         result += i
#     i += 1
# print(result)

# #3번
# i = 0
# while True :
#     i += 1
#     if i > 5 :
#         break
#     for j in range(1, i+1) :
#         print("*", end="")
#     print("")                   #줄바꿈

# #4번
# for i in range(1,101) :
#     print(i)

# #5번
# A = [70,60,55,75,95,90,80,80,85,100]
# total = 0
# for score in A :
#     total += score
# average = total/10
# print(average)

# #6번
# numbers =[1,2,3,4,5]
# result =[]
# for n in numbers :
#     if n % 2 == 1 :
#         result.append(n*2)
# print(result)
# #리스트 내포
# numbers=[1,2,3,4,5]
# result = [n*2 for n in numbers if n%2==1]
# print(result)

# #221102
# #1번
# def is_odd(number) :
#     if number%2 ==1 :   #2로 나눴을때 나머지가 1이면 홀수
#         return True
#     else :
#         return False

# a1 = is_odd(27)         #홀수 = True
# print(a1)
# a1 = is_odd(20)         #짝수 = False
# print(a1)

# #2번
# def avg_numbers(*args) :
#     result = 0
#     for i in args :
#         result += i
#     return print(result/len(args))  #매개변수를 모두 더하고, 그 수만큼 나눠서 평균구함

# avg_numbers(1,2)
# avg_numbers(1,2,3,4,5)
# avg_numbers(90,85,80)

# #3번
# # input1 = int(input("첫번째 숫자를 입력하세요: "))
# # input2 = int(input("두번째 숫자를 입력하세요: "))
# # total = input1 + input2
# # print("두 수의 합은 %s 입니다." %total)

# #4번
# print("you" "need" "python")
# print("you"+"need"+"python")
# print("you", "need", "python")
# print("".join(["you", "need", "python"]))

#180p
#5번
# f1 = open("test.txt", "w")
# f1.write("Life is too short")
# f1.close()

# f2 = open("test.txt", "r")
# print(f2.read())
# f2.close()

#6번
# user_input = input("저장할 내용을 입력하세요: ")
# f = open("test.txt", "a")
# f.write(user_input)
# f.write("\n")           #줄단위 구분위해 줄바꿈 문자 삽입
# f.close()

#7번
# f = open("test.txt", "r")   #test.txt파일의 내용을 읽기모드로 열기
# body = f.read()             #전체 내용을 읽어서 body에 저장
# f.close()
# #body에서 King kong -> Ding dong 으로 바꾸기
# body = body.replace("King kong", "Ding dong")

# f = open("test.txt", "a")   # test.txt파일을 수정모드로 열기
# f.write(body)               # body를 입력하기
# f.close()

#221104
#262p
#1번
# class Calculator:
#     #생성자
#     def __init__(self) :
#         self.value =0

#     def add(self, val):
#         self.value += val
        
#     #자식클래스         (부모클래스)                       
# class UpgradeCalculator(Calculator):
#     def minus(self, val):
#         self.value -= val

# cal = UpgradeCalculator()
# cal.add(10)
# cal.minus(7)

# print(cal.value)

# #2번
#     #자식클래스          (부모클래스)  
# class MaxLimitCalculator(Calculator):
#     #부모클래스의 add함수를 발전시키자.
#     def add(self, val) :
#         self.value += val
#         #self.value값이 100보다 크면
#         if self.value >= 100 :
#             #100으로 저장
#             self.value = 100         
#         return self.value

# cal = MaxLimitCalculator()
# cal.add(50) # 0 + 50 = 50
# cal.add(60) #50 + 60 = 110

# print(cal.value)    #100출력

# #221108
# #263p
# #3번
# print(all([1, 2, abs(-3)-3]))   #0이 있으면 False
# print(chr(ord('a')) == 'a')     #아스키로 바꿨다가(ord) 다시 문자열로 바꿈(chr)

# #4번
# def Q4(l) :
#     return l > 0

# print(list(filter(Q4, [1, -2, 3, -5, 8, -3])))

# #5번
# h = hex(234)
# print(h)

# i = int("0xea", 16)
# print(i)

# #6번
# def Q6(x) :
#     return x*3

# print(list(map(Q6, [1,2,3,4])))

# #7번
# Q7 = [-8, 2, 7, 5, -3, 5, 0, 1]
# print(max(Q7))
# print(min(Q7))

# #8번
# print(17/3)
# print(round(17/3, 4))

#221114
#322p
#1번
q1 = "a:b:c:d"
print("#".join(q1.split(":")))
#2번
q2 = {"A":90, "B":80}
print(q2.get("c",70))
#3번
q3_1 = [1,2,3]
q3_1 = q3_1 + [4,5]
print(q3_1)
q3_2 = [1,2,3]
q3_2.extend([4,5])
print(q3_2)
#4번
q4 = [20,55,67,82,45,33,90,87,100,25]
result_q4 = 0
while q4:
    ok = q4.pop()
    if ok >= 50 :
        result_q4 += ok
print(result_q4)
#5번
def fib(n) :
    if n == 0 : return 0
    if n == 1 : return 1
    return fib(n-2) + fib(n-1)

#6번
q6 = input("숫자를 입력하세요 > ")
num6 = q6.split(",")
total = 0
for n in num6 :
    total += int(n)
print(total)

#7번
def q7(n) :
    result_q7=[]
    i = 1
    while i < 10 :
        result_q7.append(n*i)
        i = i+ 1
    return result_q7   
    #타입 변환 필요
gu = int(input("한줄 구구단 숫자 입력하기 >"))
print(q7(gu))
#8번

#9번

#10번