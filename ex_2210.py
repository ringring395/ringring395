#1. 홍길동씨의 과목별 점수는 각각 다음과 같다
#국어=80
#영어=75
#과학 =55

#홍길동씨의 평균점수
sum=80+75+55
print(sum)
avg = (sum/3)
print(avg)


#2. 자연수 13이 홀수인지 짝수인지 판별해보자
#짝수면 0, 홀수면 1을 출력
a = 13
print(a%2)

#221025
num = "021025-3xxxxxx"      #주민등록번호 저장 변수
birth = num[:6]             #슬라이싱 : 인덱스 0~5     
gender = num[7]             #인덱싱 : 인덱스 7
print("년월일 : "+birth)
print("성별 : "+gender)
#포매팅코드
print("년월일 : %s" %birth) 
print("성별 : %s" %gender) 
#format 함수 
print("년월일 : {0}" .format(birth)) 
print("성별 : {0}" .format(gender))  
#추가 변수 선언 없이 바로
print("년월일 : %s" %num[:6])
print("성별 : %s" %num[7])

#113페이지
#5번
a = "a:b:c:d"
b = a.replace(":", "#")
print("정렬 전 : "+a)
print("정렬 후 : "+b)

#6번
a=[1,3,5,4,2]
a.sort()                #오름차순 정렬
print(a)     
a.sort(reverse=False)   #오름차순 정렬
print(a)     
a.sort(reverse=True)    #내림차순 정렬
print(a)     

#7번
a = ["Life", "is", "too", "short"]
result = " ".join(a)
print(result)

#8번
a = (1, 2, 3)
#a = a+(4)       #에러
a = a+(4,)
print(a)

#9번
a = dict()
a['name'] = 'pyhon'
a[('a')] = 'python'
#a[[1]] = 'python'
a[250] = 'python'
print(a)

#10번
a = {'A':90, 'B':80, 'C':70}
print(a.get('B'))       #get 함수
print(a.pop('B'))       #pop 함수
print(a)                #pop으로 B는 삭제됨

#11번
a = [1,1,1,2,2,3,3,3,4,4,5] #리스트형
aSet = set(a)               #리스트형 -> 집합형 (중복제거 위해)
print(aSet)
b = list(aSet)              #집합형 -> 리스트형 (원래 자료형으로 변환)
print(b)

#12번
a = b = [1,2,3]
a[1] = 4
print(a)
print(b)
print(id(a))
print(id(b))