#while문
kick =0
while kick < 10 :
    kick = kick + 1
    print("%d번 찼습니다." %kick)    #포매팅 (점프투파이썬57p)
    if kick == 10 :
        print("king kong kick the drum")

#키보드 입력받기
prompt = """

    1. Add
    2. Del
    3. List
    4. Quit

    Enter number : 
"""    
number = 0
while number !=4 :  #키보드로 입력한(number)가 4가 아닐 동안 반복
    print(prompt)   #prompt 문자열 출력
    number = int(input())   #input함수를 통해 사용자가 키보드로 입력한 후 숫자로 변환

print(bool(0))
print(bool(1))
print(bool(-2))
print(bool(300))

floor = 10
man = 50
while man : 
    print("내려갑니다.")
    floor = floor -1
    print("현재 %d층 입니다." %floor)
    if floor ==0 :
        print("로비에 도착했습니다.")
        break

a = 0
while a < 10 :
    a = a+1
    if a % 2 == 0 :
        continue
    print(a)

while True :
    print("무한루프")

