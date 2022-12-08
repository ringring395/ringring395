#주어진 자연수가 홀수인지 짝수인지 판별해주는 함수.
def is_odd(num):
    if num%2==1:
        print("홀수")
    elif num%2==0:
        print("짝수")

#출력 확인
is_odd(200)
is_odd(135)