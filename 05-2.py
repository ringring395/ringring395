#mod1.py파일 불러오기
#import 모듈명
#from 모듈명 import 모듈함수
# from mod1 import add
# from mod1 import sub
# from mod1 import *

#mod1.py파일에 있는 
# add 함수와
#sub함수를 호출

# add(10, 3)
# sub(10, 3)

import mod2
result1 = mod2.add(3,4)
result2 = mod2.Math()
result3 = mod2.add(mod2.pi, 10)

print(result1)
print(result2.solv(2))
print(result3)