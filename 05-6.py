#라이브러리_sys
# import sys
# print(sys.argv)
# print(sys.path)
# sys.exit()

# import pickle
# f = open("test.txt", "wb")
# data = {1:"king", 2:"sing"} #딕셔너리 자료형
# pickle.dump(data, f)        #data 딕셔너리 객체를 그대로 파일에 저장하는 기능
# f.close()

# f = open("test.txt", "rb")
# data = pickle.load(f)       #저장한 data딕셔너리 객체를 불러오는 기능
# print(data)
# f.close()

# import os
# print(os.environ)
# print(os.getcwd())
# print(os.system("dir"))
# f=os.popen("dir")
# print(f.read())

# import shutil
# shutil.copy("test.txt", "test_1.txt")

# import glob
# print(glob.glob("D:/01-STUDY/python/*test*"))

# import tempfile
# filename = tempfile.mkstemp
# print(filename)
# f = tempfile.TemporaryFile()
# print(f)
# f.close()

# import time
# print(time.time())
# print(time.localtime(time.time()))
# print(time.asctime(time.localtime(time.time())))
# print(time.ctime())
# print(time.strftime("%x", time.localtime(time.time())))
# for i in range(10):
#     print(i)
#     time.sleep(1)

# import calendar
# print(calendar.calendar(2023))
# print(calendar.prmonth(2023,1))
# print(calendar.weekday(2023,1,10))  #2023-01-10은 화요일이다.
# print(calendar.monthrange(2023,2))  #2023-02의 1일은 수요일이고, 28일까지 있다.

# import random
# print(random.random())
# print(random.randint(1, 100))   #1~100사이의 랜덤 숫자(정수형)

# # def randomPop(data) :
# #     num = random.randint(0, len(data)-1)
# #     return data.pop(num)

# def randomPop(data) :
#     num2 = random.choice(data)
#     data.remove(num2)
#     return num2

# if __name__ == "__main__" :     #java : public void static main(){}
#     data = [1, 2, 3, 4, 5]
#     while data:
#         print(randomPop(data))

# sh=[11, 22, 33, 44, 55, 66, 77]
# random.shuffle(sh)
# print(sh)
# print(random.randint(1, 45))

import webbrowser
url = "http://www.google.com"
chrome_path = "C:/program Files/Google/Chrome/Application/chrome.exe %s"
# webbrowser.get(chrome_path).open(url)
#브라우저 실행중일때, 새창으로 열림
webbrowser.get(chrome_path).open_new("http://naver.com")