#for 문
#리스트/튜플/문자열 자료형
#for 변수명 in 리스트(or 튜플, 문자열)
test_list = ["one", "two", "three"]
for i in test_list :
    print(i)

a = [(1,2), (3,4), (5,6)]
for(first,last) in a :
    print(first, last, first+last)

#for문 응용
tests=[90,25,67,45,80]      # 학생 5명의 시험 점수
student = 0                 #학생 번호
for test in tests :
    student = student +1    #학생 번호 매기기
    # 60점 넘으면 합격
    if test < 60 :
        continue        
    print("%d번 학생 : 합격" %student)    

#for문과 함께 자주 사용하는 range함수
a = range(10)
print(a)

b = range(1,11)
print(b)

add = 0
for i in range(1,11) :
    add = add + i
    print(add)
print("총 합산 : %d" %add)

#구구단
#2~9단
for i in range(2,10) :
    print("%d단" %i)
    #1~9
    for j in range(1,10) :
        print(i*j, end=" ") #한줄에 한단 출력
    print("")               #단 별로 줄바꿈   

a = [1,2,3,4]
result = []
for num in a :
    result.append(num*3)
print(result)

a = [1,2,3,4]
result = [num*3 for num in a]
print(result)