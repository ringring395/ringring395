#패키지 안의 함수 실행하기
#1. 모듈을 import하여 실행
# import dynamite.ding.dong
# dynamite.ding.dong.dong_test()  #Ding dong

# import dynamite.king.kong
# print(dynamite.king.kong.kong_test)  #King kong

#2. 모듈이 있는 디렉토리까지를 from...import 하여 실행
# from dynamite.ding import dong
# dong.dong_test()     #Ding dong

# from dynamite.king import kong
# print(kong.kong_test)     #King kong

#3. 모듈의 함수를 직접 import하여 실행
# from dynamite.ding.dong import dong_test
# dong_test()     #Ding dong

# from dynamite.king.kong import kong_test
# print(kong_test)     #King kong

# from dynamite.ding import *
# dong.dong_test()

# from dynamite.king import *
# print(kong.kong_test)

from dynamite.ding.dong import dong_test

dong_test()

