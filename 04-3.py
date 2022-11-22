#파일 읽고 쓰기
#1. 파일 생성하기
# f = open("d://새파일.txt", "w")
# #파일에 내용 입력
# for i in range(1,11) :
#     data = "%d만원 입니다.\n" %i
#     f.write(data)       #data변수의 내용을 파일에 입력

# f.close()

#3. 파일에 저장된내용을 python으로 읽어들이는 방법
#3-1. readline함수
# f = open("D://새파일.txt", "r")     # r : 읽기모드
# read1 = f.readline()
# print(read1)
# f.close()

#전체를 출력해보자.
# f = open("D://새파일.txt", "r")
# while True :        #무한루프
#     read1_2 = f.readline()
#     if not read1_2 :   #line변수에 아무것도 없으면
#         break       #반복문 취소
#     print(read1_2)
# f.close()

#3-2. readlines함수
# f = open("D://새파일.txt", "r")     # r : 읽기모드
# read2 = f.readlines()
# print(read2)
# f.close()
#3-3. read함수
# f = open("D://새파일.txt", "r")
# read3 = f.read()
# print(read3)
# f.close()

#4. 파일에 기존내용 그대로 두고, 새로운 내용 추가하기
f = open("D://새파일.txt", "a")
for i in range(11,21) :
    data = "%d만원 입니다.\n" %i
    f.write(data)
f.close()