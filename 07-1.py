#정규식 사용 선언   (data 앞에 선언해준다)
import re
#주어진 주민번호 예시
data = """
park 021111-3000000
kim  041027-4000000
"""
#이중 for문
result =[]
for line in data.split("\n") :      #엔터 기준으로 문자 나눠서 line에 저장
    word_result =[]
    for word in line.split(" ") :   #공백 기준으로 문자 나눠서 word에 저장
        #     길이 14     &&  word 첫6글자가 숫자냐 && word 인덱스7부터 끝까지 숫자냐
        if len(word) == 14 and word[:6].isdigit() and word[7:].isdigit() :
            #첫6글자랑 -******* 결합해서 word에 다시 넣어라
            word = word[:6] +"-*******"
        #word값을 word_result에 추가해라
        word_result.append(word)
    #word_result랑 공백이랑 join해서 result에 추가해라
    result.append(" ".join(word_result))
#result랑 엔터랑 join해서 출력해라
print("\n".join(result))

#정규식 적용
#   \d : 숫자
#   {n} : n만큼 반복
            #숫자6개반복 - 숫자7개반복
reg = re.compile("(\d{6})[-]\d{7}")
#   \g<ring> : ring 이름으로 그룹화
print(reg.sub("\g<1>-*******", data))
