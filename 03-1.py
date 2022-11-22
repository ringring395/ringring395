#if문
hello = True
if hello:
    print("It's me")
else:
    print("He's gone")   
    print("T.T") 
print("Bye")

#비교 연산자
#돈이 5000 이상 있거나 카드가 있으면 butter를 사고,
#아니면 전화를 해라
money = 5000
card = True
if money>=5000 or card :
    print("Buy Butter")
else :
    print("Call me")

# x in s, x not in s
print(1 in [1,2,3])             #리스트 / True
print("a" in ("a", "b", "c"))   #튜플 / True
print("k" in "Kingkong")        #문자열 / True

print(1 not in [1,2,3])             #리스트 / False
print("d" not in ("a", "b", "c"))   #튜플 / True
print("s" not in "Kingkong")            #문자열 / True

#주머니에 돈이 있으면 커피랑 케이크 주문하고, 그렇지 않으면 커피만 마셔라
pocket = ["cellphone", "pen", "money", "lipstick"]
if "money" in pocket :
    print("Order coffee & cake^^")
else :
    print("Order only coffee")

#주머니에 돈이 있으면 가만히 있고, 그렇지 않으면 커피를 마셔라
pocket = ["cellphone", "pen", "money", "lipstick"]
if "money" in pocket :
    pass
else :
    print("Order only coffee")

#elif
bag = ["key", "cellphone"]
card = True
#가방에 돈이 있으면 커피를 주문하고,
if "money" in bag :
    print("Order only coffee")
#가방에 돈은 없지만 카드가 있으면 커피랑 케이크를 주문하고,
elif card :
    print("Order coffee & cake :)") 
#둘다 없으면 전화해라.
else :
    print("call me")

#if문
score = 70
if score >=60 :
    msg = "success"
else : 
    msg = "failure"
print(msg)

#조건부 표현식
#참일때 결과 if 조건문 else 거짓일때 결과
message = "success" if score >=60 else "failure"
print(message)
