#불 자료형(boolean)
#python에서는 첫글자가 대문자 T, F임을 주의!
a = True
b = False

print(1==1) #True
print(2>1)  #True
print(2<1)  #False

a=[] #list 자료형, 값 없음(False)
if a:
    print("참")
else:
    print("거짓")    
a=[1,2,3]   #list 자료형, 값 있음(True)
if a:
    print("참")
else:
    print("거짓")

print(bool(""))           #False
print(bool("butter"))     #True
print(bool([]))           #False
print(bool(["king", "kong"]))     #True