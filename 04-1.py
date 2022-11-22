#함수def
#두 수를 더하는 add 함수 선언
from re import A


def add(a, b) :
    return a+b
#두 수를 더하는 add 함수 호출
c = add(3, 4)
print(c)

#함수 선언 (*매개변수)
def add_many(*args) :
    result = 0
    for i in args :
        result = result+i
    return result

#함수 호출 (*매개변수)
result = add_many(1,2,3)
print(result)
result = add_many(1,2,3,4,5,6,7,8,9,10)
print(result)

#사용자가 + 선택하면 값들을 더하고,
# * 선택하면 값들을 곱하라.
#함수 선언
def add_mul(choice, *num) :
    if choice == "+" :      #사용자가 + 를 선택하면
        result = 0
        for i in num :      #값들을 더하라
            result += i
    elif choice == "*" :    #사용자가 * 를 선택하면
        result = 1          #곱하기니까 기본값 1로 준다.
        for i in num :      #값들을 곱하라
            result *= i
    return result

#함수 호출
result = add_mul("+", 1,2,3,4,5)
print(result)
result = add_mul("*", 1,2,3,4,5)
print(result)

# 리턴 여러개
def add_and_mul(a, b) :
    return a+b, a*b     #리턴 2개

result = add_and_mul(3, 4)
print(result)           #튜플 자료형 하나로 출력

def add_and_mul1(a, b) :
    return a+b      #실행O
    return a*b      #실행X

result = add_and_mul1(3, 4)
print(result) 

#매개변수 초기화
def myself(name, age, man=True) :
    print("나의 이름은 %s입니다." %name)
    print("나이는 %d살 입니다." %age)
    if man :
        print("남자입니다.")
    else :
        print("여자입니다.")

# 호출
myself("임영웅", 30)
myself("임영웅", 30, True)
myself("송가인", 30, False)

#변수의 범위
a = 1
def vartest(a) :
    a = a+1
    print(a)        #지역변수 a 출력

vartest(a)
print(a)            #전역변수 a 출력

#return활용
a = 1
def vartest_re(a) :
    a = a+1
    return a

a = vartest_re(a)
print(a)

#global활용
a = 1
def vartest_gl() :
    global a
    a = a+1

vartest_gl()
print(a)

# 일반적인 함수 표현
def add1(a, b) :
    return a+b

result = add1(3, 4)
print("일반 함수 : %d" %result)

#lambda 함수 표현
#lambda 매개변수 : 매개변수 표현식
addLambda = lambda a, b : a+b
result = addLambda(3,4)
print("lambda 함수 : %d" %result)