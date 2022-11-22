#ㅇㅖ외처리
# f = open("나에게없는파일", "r")    #FileNotFoundError
# print(4/0)                        #ZeroDivisionError
# a = [1,2,3]
# print(a[4])                         #IndexError

try :
    4/0
except ZeroDivisionError as e :
    print(e)


# try:
#     #무언가를 수행한다.
#     f = open("foo.txt", "w")
# finally :
#     f.close()

#오류 여러개
try :
    print(4/0)          #ZeroDivisionError
    a=[1,2,3]
    print(a[4])         #IndexError
except ZeroDivisionError :
    print("오류 여러개 : 0으로 나눌 수 없습니다.")
except IndexError :
    print("오류 여러개 : 인덱싱할 수 없습니다.") 

#오류 여러개(괄호로 묶기)
try :
    print(4/0)          #ZeroDivisionError
    a=[1,2,3]
    print(a[4])         #IndexError
except (ZeroDivisionError, IndexError) as e:
    print("오류 여러개(괄호묶기) : %s" %e)

#오류 회피하기
try :
    f = open("나에게없는파일", "r")
except FileNotFoundError :
    pass

#오류 강제 발생222
class Bird :
    def fly(self) :
        raise NotImplementedError

#자식클래스(부모클래스) -> 상속관계
class Eagle(Bird) :
    def fly(self) :     #오버라이딩
        print("very fast")

eg = Eagle() #java : Eagle eg = new Eagle()
eg.fly()

#예외 만들기
class MyError(Exception) :  #상속관계
    def __str__(self) :     #자식클래스에 메세지 심기
        return "허용되지 않는 별명입니다."

def say_nick(nick) :
    if nick == "바보" :
        raise MyError()     #오류 강제 발생
    print(nick)

#예외 처리 적용
try :
    say_nick("천사")
    say_nick("ringring")
    say_nick("바보")
except MyError as e:
    #print("허용되지 않는 별명입니다.")
    print(e)