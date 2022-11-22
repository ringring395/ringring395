import re
#메타문자
# #문자클래스
# chr = re.compile("[abc]")  # [abc] = [a-c] : a부터 c까지
# print(chr.match("a"))      # <re.Match object; span=(0, 1), match='a'>
# print(chr.match("before")) # <re.Match object; span=(0, 1), match='b'>
# print(chr.match("hello"))  # None
# #dot
# dot = re.compile("a.b")
# print(dot.match("aab"))     # <re.Match object; span=(0, 3), match='aab'>
# print(dot.match("a5b"))     # <re.Match object; span=(0, 3), match='a5b'>
# print(dot.match("a=b"))     # <re.Match object; span=(0, 3), match='a=b'>
# print(dot.match("abc"))     # None
# # *
# repeat = re.compile("ki*g")
# print(repeat.match("kg"))       # <re.Match object; span=(0, 2), match='kg'>
# print(repeat.match("kig"))      # <re.Match object; span=(0, 3), match='kig'>
# print(repeat.match("kiiiiig"))  # <re.Match object; span=(0, 7), match='kiiiiig'>
# # +
# repeat2 = re.compile("ki+g")
# print(repeat2.match("kg"))       # None
# print(repeat2.match("kig"))      # <re.Match object; span=(0, 3), match='kig'>
# print(repeat2.match("kiiiiig"))  # <re.Match object; span=(0, 7), match='kiiiiig'>
# # { }
# repeat3 = re.compile("ki{2,5}g")     # i가 2번 반복되면 매칭
# print(repeat3.match("kg"))       # None
# print(repeat3.match("kig"))      # None
# print(repeat3.match("kiiiiig"))  # <re.Match object; span=(0, 7), match='kiiiiig'>
# # ? 
# repeat4 = re.compile("ki?g")     # i가 0~1번 반복되면 매칭
# print(repeat4.match("kg"))       # <re.Match object; span=(0, 2), match='kg'>
# print(repeat4.match("kig"))      # <re.Match object; span=(0, 3), match='kig'>
# print(repeat4.match("kiiiiig"))  # None
#match 함수
# p = re.compile("[a-z]+")    # 알파벳(a~z)[a-z]가 1번이상(+) 반복되면 매칭
# print(p.match("ringring"))  # <re.Match object; span=(0, 8), match='ringring'>
# print(p.match("7ringring")) # None
# #if문으로 확인해보기
# m = p.match("dingdong")
# if m :
#     print("Matching 성공 :)")
# else :
#     print("Matching 실패")
# #search함수
# print(p.search("ringring"))     # <re.Match object; span=(0, 8), match='ringring'>
# print(p.search("7ringring"))    # <re.Match object; span=(1, 9), match='ringring'>
# #findall함수
# print(p.findall("king kong kick the drum"))
# print(p.findall("rolling on like a rolling stone"))
# print(p.findall("rolling on like 101 rolling stone"))
# #finditer함수
# result1 = p.finditer("rolling on like a rolling stone")
# result2 = p.finditer("rolling on like 101 rolling stone")
# for r in result1 :
#     print(r)
# for r in result2 :
#     print(r)
# #match의 함수
# p = re.compile("[a-z]+")
# m = p.match("butter")
# print(m.group())    # 매칭된 문자열을 돌려줌
# print(m.start())    # 매칭된 문자열의 시작인덱스
# print(m.end())      # 매칭된 문자열의 끝인덱스
# print(m.span())     # 매칭된 문자열의 시작위치와 끝위치를 튜플로 돌려줌
# #search함수
# s = p.search("7 butter")
# print(s.group())    # 매칭된 문자열을 돌려줌
# print(s.start())    # 매칭된 문자열의 시작인덱스
# print(s.end())      # 매칭된 문자열의 끝인덱스
# print(s.span())     # 매칭된 문자열의 시작위치와 끝위치를 튜플로 돌려줌
# 반복 ( . )
p = re.compile("a.b")   # a로 시작하고 b로 끝나는 문자(중간에는 줄바꿈 제외, 어떤 문자도 ok)
print(p.match("a0b"))     # <re.Match object; span=(0, 3), match='a0b'>
print(p.match("a\nb"))    # None
#DOTALL
p1 = re.compile("a.b", re.DOTALL)
print(p1.match("a0b"))      # <re.Match object; span=(0, 3), match='a0b'>
print(p1.match("a\nb"))     # <re.Match object; span=(0, 3), match='a\nb'>
#정규식 : 소문자만
i1 = re.compile("[a-z]")
print(i1.match("dingdong")) # <re.Match object; span=(0, 1), match='d'>
print(i1.match("Dingdong")) # None
print(i1.match("DINGDONG")) # None
#IGNORECASE
i2 = re.compile("[a-z]", re.IGNORECASE)
print(i2.match("dingdong")) # <re.Match object; span=(0, 1), match='d'>
print(i2.match("Dingdong")) # <re.Match object; span=(0, 1), match='D'>
print(i2.match("DINGDONG")) # <re.Match object; span=(0, 1), match='D'>
#정규식 : king으로 시작해야함
m1 = re.compile("^king\s\w+")
data = """king Kong kick the drum
rolling on like a rolling stone
king of the word
Ding dong call me on my phone
king of the king
"""
print(m1.findall(data))
#MULTILINE
m2 = re.compile("^king\s\w+", re.MULTILINE)
data = """king Kong kick the drum
rolling on like a rolling stone
king of the word
Ding dong call me on my phone
king of the king
"""
print(m2.findall(data))
#한줄로 입력한 정규식
x1 = re.compile(r'&[#](0[0-7]+|[0-9]+|x[0-9a-fA-F]+);')
#VERBOSE
x2 = re.compile(r"""
&[#]                #
(
    0[0-7]+         #8진수 정규식
    |[0-9]+         #10진수 정규식
    |x[0-9a-fA-F]+  #16진수 정규식
);
""", re.VERBOSE)

#역슬러시\section 형태의 정규식을 만들고 싶다면????
#역슬러시 하나\
b = re.compile("\section")  # \s 정규식의 ection : [\t\n\r\f\v]ection으로 오인
#역슬러시 둘\\
b2 = re.compile("\\section")    # \section형태의 정규식
#r : 하나만 써도 2개 쓴것과 동일
b22 = re.compile(r"\section")    # \section형태의 정규식
