#커피 자판기
coffee = 10     #커피의 재고
while True :
    money = int(input("돈을 넣어주세요 : "))
    if money == 500 :
        print("커피 제공")
        coffee = coffee -1
    elif money > 500 :
        print("거스름돈 : %d / 커피 제공" %(money-500))
        coffee = coffee -1
    else :
        print("돈 반환 / 커피 제공 X")
        print("%d잔 남았습니다." %coffee)
    if coffee == 0 :
        print("커피 재고 없음 / 판매 중지")
        break