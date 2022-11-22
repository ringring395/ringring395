# result = 0

# def add(num):       #add함수 선언
#     global result   #전역변수
#     result += num
#     return result

# print(add(3))       #add함수 호출
# print(add(4))       #add함수 호출


# class Calculator :      #class선언
#     def __init__(self): #생성자
#         self.result = 0
        
#     def add(self,num) : #add함수 선언
#         self.result += num
#         return self.result

# cal1 = Calculator()     #class호출
# cal2 = Calculator()     #class호출

# print(cal1.add(3))      #add함수 호출
# print(cal1.add(4))      #add함수 호출

# print(cal2.add(3))      #add함수 호출
# print(cal2.add(7))      #add함수 호출

#188p
#계산기 클래스 만들기
class FourCal :         #FourCal 클래스 선언
    #생성자
    def __init__(self) :
        self.first = 0
        self.second = 0

    #두 수를 입력받는 기능 setdata 선언
    def setdata(self, first, second) :
        self.first = first
        self.second = second

    #더하기 기능 add 선언
    def add(self) :
        result = self.first + self.second
        return result    

    #빼기 기능 sub 선언
    def sub(self) :
        result = self.first - self.second
        return result    

    #나누기 기능 div 선언
    def div(self) :
        result = self.first / self.second
        return result  

    #곱하기 기능 mul 선언
    def mul(self) :
        result = self.first * self.second
        return result  

#FourCal 클래스 호출(FourCal 인스턴스화) :FourCal fc = new FourCal();
a = FourCal() 
#setdata함수를 이용 : first에는 40, second에는 20으로 초기화      
a.setdata(40,2)

print(a.first)      #setdata의 first값 확인
print(a.second)     #setdata의 second값 확인
#FourCal클래스의 add함수호출 : setdata의 first와 second의 더하기
print(a.add()) 
#FourCal클래스의 sub함수호출 : setdata의 first와 second의 빼기     
print(a.sub())      
#FourCal클래스의 div함수호출 : setdata의 first와 second의 나누기
print(a.div())      
#FourCal클래스의 mul함수호출 : setdata의 first와 second의 곱하기
print(a.mul())      

#class 상속
#FourCal 클래스 - 부모클래스
#MoreFourCal 클래스 - 자식클래스
#MoreFourCal 클래스는 FourCal클래스로부터 상속 받음
#java : MoreFourCal() extends FourCal
class MoreFourCal(FourCal) :        #상속 : 자식클래스(부모클래스)
    def power(self) :
        result = self.first ** self.second  #거듭제곱
        return result

    #나누기 기능 div 선언 - 오버라이딩
    def div(self) :
        if self.second == 0 or self.first == 0 :
            return 0
        else :
            result = self.first / self.second
            return result

mfc = MoreFourCal()
mfc.setdata(40, 2)
print(mfc.first)
print(mfc.second)
print(mfc.add())
print(mfc.sub())
print(mfc.div())
print(mfc.mul())
print(mfc.power())

mfc.setdata(40, 0)
print(mfc.div())

#클래스 변수

class Family :
    lastname = "김"

print(Family.lastname)
a = Family()    #Family a = new Family()
b = Family()    #Family b = new Family()
print(a.lastname)
print(b.lastname)