#a = input("숫자를 입력하세요 >") #사용자가 키보드로 입력
#print(a)    # 입력된 값을 출력

a = 123
print(a)
b = "Butter"
print(b)
c = [11,22,33]
print(c)

print("King" "kong" "kick" "the drum")
print("King"+"kong"+"kick"+"the drum")
print("King", "kong", "kick", "the drum")

for i in range(10) :
    print(i)            #한줄에 i 하나씩
for i in range(10) :    
    print(i, end="")    #한줄에 전체 출력(공백X)
print("")
for i in range(10) :    
    print(i, end=" ")   #한줄에 전체 출력(공백O)