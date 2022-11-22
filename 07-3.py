#309p
#메타문자
import re
              # Crow 또는 Servo 단어
p = re.compile("Crow|Servo")
m = p.match("CrowHello")
print(m)      # <re.Match object; span=(0, 4), match='Crow'>
#^
p2 = re.search("^king","king kong kick the drum")
print(p2)       #<re.Match object; span=(0, 4), match='king'>
p2 = re.search("^kong","king kong kick the drum")
print(p2)       #None
#$
p3 = re.search("drum$", "king kong kick the drum")
print(p3)       # <re.Match object; span=(19, 23), match='drum'>
p3 = re.search("butter$", "king kong kick the drum")
print(p3)       #None

data = """king kong kick the drum
king! of the king
king@ of the world"""

#\A
p4 = re.compile("^king",re.MULTILINE)   #여러줄 적용
p4a = re.compile("\Aking",re.MULTILINE)   #여러줄 적용
print(p4.findall(data)) #['king', 'king', 'king']
print(p4a.findall(data)) #['king']
#\Z
p5 = re.compile("drum$", re.MULTILINE)   #여러줄 적용
p5z = re.compile("drum\Z", re.MULTILINE)   #여러줄 적용
print(p5.findall(data)) #['drum']
print(p5z.findall(data)) #[]
#\b
p6 = re.compile(r"\bclass\b")
print(p6.search("no class at all")) #<re.Match object; span=(3, 8), match='class'>
print(p6.search("the declassified algorithm"))  #None
#\B
p6 = re.compile(r"\Bclass\B")
print(p6.search("no class at all")) #None
print(p6.search("the declassified algorithm"))  #<re.Match object; span=(6, 11), match='class'>
#그루핑
p11 = re.compile("(ABC)+")
m11 = p11.search("ABCABCABC OK?")
print(m11)            #<re.Match object; span=(0, 9), match='ABCABCABC'>
print(m11.group(0))   #ABCABCABC
print(m11.group(1))   #ABC
                # 문자 공백 숫자 - 숫자 - 숫자
                # (그룹1)  ((그룹3)  그룹2    )
p22 = re.compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
m22  = p22.search("king 010-0000-0000")
print(m22)          #<re.Match object; span=(0, 18), match='king 010-0000-0000'>
print(m22.group(1)) #king
print(m22.group(2)) #010-0000-0000
print(m22.group(3)) #010
#재참조
                #  (그룹) 공백 그룹인덱스번호
p33 = re.compile(r"(\b\w+)\s+\1")
m33 = p33.search("king kong kick the the drum").group()
print(m33)         #the the
#그루핑 네이밍
                # (그룹1)  ((그룹3)  그룹2    )
p44 = re.compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
                # (네이밍: 그룹1)  ((그룹3)  그룹2    )
p44 = re.compile(r"(?P<name>\w+)\s+((\d+)[-]\d+[-]\d+)")
m44 = p44.search("king 010-0000-0000")
print(m44)          # <re.Match object; span=(0, 18), match='king 010-0000-0000'>
print(m44.group(1))         #king
print(m44.group("name"))    #king

                #   (그룹)        공백  그룹네이밍재참조
p33n = re.compile(r"(?P<word>\b\w+)\s+(?P=word)")
m33n = p33n.search("king kong kick the the drum").group()
print(m33n)         #the the
#전방탐색
    # .(모든문자) +(반복)
p55 = re.compile(".+:")
m55 = p55.search("http://google.com")
print(m55.group())
#긍정형
    # .(모든문자) +(?=반복)
pposi = re.compile(".+(?=:)")
mposi = pposi.search("http://google.com")
print(mposi.group())
#부정형
    # .(모든문자) +(?!반복)
pnega = re.compile(".+(?!:)")
mnega = pnega.search("http://google.com")
print(mnega.group())
#문자열 바꾸기
                # | : or 의미
ps = re.compile("(king|sing|ding)")
print("king kong sing a song")
#ps의 문자열을 찾아서 ring으로 바꿔라.
print(ps.sub("ring", "king kong sing a song"))
print(ps.sub("ring", "king kong sing a song", count=1))
#subn
print(ps.subn("ring", "king kong sing a song"))
                #(그룹1)   ((그룹3)   그룹2   )
ps1 = re.compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
                #  (그룹1: name)   (그룹2:phone (그룹3)         )
ps1 = re.compile(r"(?P<name>\w+)\s+(?P<phone>(\d+)[-]\d+[-]\d+)")
print(ps1.sub("\g<phone> \g<name>", "king 010-0000-0000"))
#hexrepl
def hexrepl(match) :
    value = int(match.group())
    return hex(value)
pdef = re.compile(r"\d+")
print(pdef.sub(hexrepl, "I'll melt your heart into 2")) # 2 -> 0x2
                    # 2022 -> 0x7e6   14 -> 0xe 
print(pdef.sub(hexrepl, "2022 November 14th, monday"))  
