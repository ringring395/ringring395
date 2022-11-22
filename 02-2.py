#자료형_문자열
a = "Hi~ ringring"
b = 'Python is fun'
c = """Life is too short, You need python"""
d = '''Life is too short, You need python'''

print(a)
print(b)
print(c)
print(d)
#큰따옴표 안에 작은따옴표
food = "Python's favorite food is perl"
#단, 작은따옴표 안에 작은따옴표
food1 = 'Python\'s favorite food is perl'

#작은따옴표 안에 큰따옴표
say = '"Python is very easy." he says'
#단, 큰따옴표 안에 큰따옴표
say1 = "\"Python is very easy.\" he says"

print(food)
print(food1)
print(say)
print(say1)

# 줄 나눠서 출력하기
# hello~ (엔터)
# It's me

print("hello~ It's me") #한줄로 출력
print("hello~ \nIt's me") # 이스케이프코드 \n
print("""hello!
It's me""") #따옴표 3개 쓰기


# 문자열 연산하기
first = "hello"
second = " It's me"

# 문자열 + 문자열 (연결)
print(first + second)

# 문자열 * 숫자
print(first * 2)
print("="*20)
print("Ringring")
print("="*20)

print(food[4])
print(len(a))

#문자열 길이 구하기
a="hello~"
b="It's me"
print(len(a))  #6
print(len(b))   #7

#문자열 인덱싱
aa="Smooth like butter"
print(aa[5])     #h
print(aa[0])     #S
print(aa[12])    #b
# -(마이너스)는 뒤에서부터 카운팅
print(aa[-1])    #r

#문자열 슬라이싱
aa="Smooth like butter"

#like를 추출해보자.
print(aa[7:11])     #like

#처음부터 인덱스의 앞번호까지 출력
print(aa[:11])      #Smooth like

#인덱스 번호부터 마지막까지 출력
print(aa[12:])      #butter

#12부터 뒤에서 4번째의 앞번호까지 출력
print(aa[12:-4])    #bu

#오늘날짜와 네임을 연결한 문자열
a="20221025Ringring"
year = a[:4]    #2022
month = a[4:6]  #10
day = a[6:8]    #25
name = a[8:]    #Ringring
print(year)
print(month)
print(day)
print(name)


#나이를 저장하는 변수 선언
age = 28
print("나이는 %d입니다." %age)
age = 20
print("나이는 %d입니다." %age)  #포매팅코드
print("나이는 {0}입니다." .format(age)) #format함수

#커피 주문수
coffee="6잔"
print("아이스아메리카노 %s 주문을 받았다." %coffee) #포매팅코드
print("아이스아메리카노 {0} 주문을 받았다." .format(coffee)) #format 함수

#아이스크림 5개를 먹었다. 3일동안 배가 아팠다.
ice = "5개"
day = 3
print("아이스크림 %s를 먹었다. %d일동안 배가 아팠다." %(ice, day)) #포매팅코드
print("아이스크림 {0}를 먹었다. {1}일동안 배가 아팠다." .format(ice, day)) #format함수

s="안녕"
name="Ringring"
print("%s" %s)  #s 문자열을 %s에 대입
#숫자를 넣으면 숫자만큼의 사이즈를 생성해서 넣게됨
#오른쪽
print("%20s" %s)    #오른쪽 정렬(포매팅코드)
print("{0:>20}" .format(s))    #오른쪽 정렬(format 함수)
print("{0:@>20}" .format(s))    #오른쪽 정렬(format 함수) / 공백대신 @ 넣기
#가운데
print("{0:^20}" .format(s))    #가운데 정렬(format 함수)
print("{0:=^20}" .format(s))   #가운데 정렬(format 함수) / 공백대신 = 넣기
print("{0:^^20}" .format(s))   #가운데 정렬(format 함수) / 공백대신 ^ 넣기
#왼쪽
print("{0:<20}" .format(s))    #왼쪽 정렬(format 함수)
print("%-20s %s" %(s, name))   #왼쪽 정렬(포매팅코드)
print("{0:<20} {1}" .format(s, name))   #왼쪽 정렬(format 함수)

#소수점 표현
f=3.141592
print("%f" %f)     #3.141592
print("{0:f}" .format(f)) #format 함수
# 소수점 4자리 표현
print("%0.4f" %f)  #3.1416
print("{0:0.4f}" .format(f)) #format 함수
# 총 길이는 10, 소수점은 4자리 표현
print("%10.4f" %f) #    3.1416
print("{0:10.4f}" .format(f)) #format 함수

#f 문자열 포매팅
who = "King Kong"
what = "the drum"
time = 10

print(f"{who} kick {what} rolling on like a rolling stone")
print(f"I saw it {time+2} times.")

#딕셔너리
#javascript의 JSON타입이 python 딕셔너리
d={"who":"King Kong", "what":"the drum"}
print(f"{d['who']} kick {d['what']} rolling on like a rolling stone")

#f 문자열 포매팅_정렬
print(f"{'butter':$<10}")    #왼쪽 정렬 (+ $문자로 공백 채움)
print(f"{'butter':^^10}")    #왼쪽 정렬 (+ ^문자로 공백 채움)
print(f"{'butter':=>10}")    #왼쪽 정렬 (+ =문자로 공백 채움)

#f 문자열 포매팅_소수점
y=3.141592
print(f"{y:0.4f}")  #소수 넷째자리까지 표현
print(f"{y:10.4f}")  #총 10자 중에서 소수 넷째자리까지 표현
print(f"{y:*>10.4f}")  #공백은 *로 채움

#count 함수
a="butter"
print(len(a))       #문자열 길이 -- 6
print(a.count("t")) #문자열에서 "t" 갯수 --2 

#find 함수
af="King Kong kick the drum rolling on like a rolling stone"
print(af.find("d")) #문자열에서 "d"의 위치 --19
                    #0부터 시작하기 때문에 20아니라 19 출력
print(af.find("f")) #f문자열은 없으므로 -1

#index 함수     
print(af.index("d")) #문자열에서 "d"의 위치 --19
#print(af.index("f")) #f문자열 없어서 오류 발생

#join 함수
print("*".join("butter"))   #butter문자열 사이에 * 삽입
print("*".join(['b','u','t','t','e','r'])) #리스트의 인덱스 사이에 * 삽입

#upper 함수
s = "king kong"
s2 = "Butter"
print(s.upper())
print(s2.upper())

#lower 함수
s = "KING KONG"
print(s.lower())
print(s2.lower())

#공백제거
s = "         butter         "
print(s.lstrip()) #왼쪽 공백 제거(lstrip)
print(s.rstrip()) #오른쪽 공백 제거(rstrip)
print(s.strip())  #양쪽 공백 제거(strip)

#문자열 바꾸기(replace)
s = "King Kong kick the drum"
#King Kong을 Ding dong으로 바꾸기
print(s.replace("King Kong", "Ding dong"))

#문자열 나누기(split)
s = "Smooth like butter"
print(s.split())    #공백 기준 나누기
print(s.split("t")) #t 기준 나누기

s1 = "king:kong:ding:dong"
print(s1.split(":")) # : 기준 나누기
