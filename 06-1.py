#269p
#구구단
# def gu(n) : #n은 단
#     result =[]  #결과를 저장하는 list 자료형
#     result.append(  n*1  )  # 단*1
#     result.append(  n*2  )  # 단*2
#     result.append(  n*3  )  # 단*3
#     result.append(  n*4  )  # 단*4
#     result.append(  n*5  )  # 단*5
#     result.append(  n*6  )  # 단*6
#     result.append(  n*7  )  # 단*7
#     result.append(  n*8  )  # 단*8
#     result.append(  n*9  )  # 단*9
#     return result
#2단만 출력해보자.
# print(gu(2))

#구구단 수정1
# def gu1(n) :
#     result =[]
#     i = 1
#     while i < 10 :      #while문 : i는 1부터 10보다 작은수까지
#         result.append(  n*i  )
#         i += 1
#     return result
# 2단만 출력해보자.
# print(gu1(2))

#구구단 수정2
# def gu2(n) :
#     result =[]
#     i = 1
#     while i < 10 :      #while문 : i는 1부터 10보다 작은수까지
#         result.append(  n*i  )
#         i += 1
#     return result
#입력받은 숫자의 구구단을 출력해보자.
# dan = int(input("몇단을 볼까요? >"))
# print(gu2(dan))

#273p
#3과 5의 배수 합하기
# n = 1
# # for n in range(1, 1000) :     -> for문으로 표현하기
# #     print(n)
# result = 0
# while n <100 :  
#     if n%3==0 or n%5==0:     #x의배수 : x로 나눌때 나머지 0   
#        print(n)    # 1~99출력됨.
#        result += n
#     n += 1
# print(result)

# result = 0
# for n in range(1, 100) :
#     if n%3==0 or n%5==0:
#         result += n
# print(result)

#276p
#게시판페이징
#총 페이지수 = (총건수 / 한페이지당 노출되는 건수) + 1
#      1    =    5   /          10             + 1
#      2    =   15   /          10             + 1
#      4    =   30   /          10             + 1
#페이징
# def paging(total, onepage):
#     if total%onepage == 0 :
#         return total // onepage
#     else :    
#         return total // onepage + 1

# print("%d 페이지" %paging(5, 10))
# print("%d 페이지" %paging(15, 10))
# print("%d 페이지" %paging(30, 10)) 

#278p
# import sys
# option = sys.argv[1]

# # print(option)
# # print(memo)

# if option == "-a" : #입력받은 문자열을 메모장에 쓰기
#     memo = sys.argv[2]  #입력받은 문자열(king kong kick the drum)
#     f = open("simpleMemo.txt", "a") # a: 추가(수정) 모드
#     f.write(memo)
#     f.write("\n")
#     f.close()
# elif option == "-v" :   #메모장 내용 조회하기
#     f = open("simpleMemo.txt")
#     memo = f.read()
#     f.close()
#     print(memo)

#282p
#탭 -> 공백4칸
# import sys
# src = sys.argv[1]   #탭 문자열이 포함된 원본파일 original
# dst = sys.argv[2]   #탭 문자열 -> 공백4칸으로 치환된 수정파일 change

# #원본 파일을 열어라 original
# f = open(src)
# tab_content = f.read()  #원본파일 original의 내용을 읽어서 tab_content에 저장
# print(tab_content)
# f.close()
# #치환(탭 -> 공백4칸으로 바꾸고 space_content에 저장)
# # space_content = tab_content.replace("\t", "    ")     아래와 동일
# space_content = tab_content.replace("\t", " "*4)
# f = open(dst, "w")      #수정파일 change에 내용 쓰기모드로 열기
# f.write(space_content)  #공백4칸으로 치환된 내용 쓰기
# f.close()
# print(space_content)

#286p
# import os
# #search함수 선언
# def search(dir_main) :      #dir_main : 시작 디렉터리
#     print(dir_main)
#     filenames = os.listdir(dir_main)    #dir_main 디렉터리 안의 파일+폴더들을 리스트로 저장
#     for filename in filenames :
#         full_name = os.path.join(dir_main, filename)    #디렉터리명, 파일명 join하기
#         ext = os.path.splitext(full_name)              #splitext : 파일명, 확장자명 잘라줌 
#         print(ext)                      #파일명, 확장자명 잘린채로 확인
#         if ext[-1] == ".txt" :          #[-1] : 뒤에서 첫번째
#             print(full_name)
#     # print(filenames)
# #search함수 호출
# search("D:/")

import os
#search함수 선언
def search(dir_main) :      #dir_main : 시작 디렉터리
    try:
    #dir_main 디렉터리 안의 파일+폴더들을 리스트로 저장    
        filenames = os.listdir(dir_main)    
        for filename in filenames :
            full_name = os.path.join(dir_main, filename)    #디렉터리명, 파일명 join하기
            if os.path.isdir(full_name) :       #하위 디렉터리가 있으면,
                search(full_name)               #해당 폴더 다시 검색
            else : 
                ext = os.path.splitext(full_name)[-1]           #splitext : 파일명, 확장자명 잘라줌 
                if ext == ".txt" :                              #[-1] : 뒤에서 첫번째
                    print(full_name)
    except PermissionError :
        pass
#search함수 호출
search("D:/")
