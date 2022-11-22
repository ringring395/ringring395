import numpy as np
#               15개        3행, 5열    
arr1 = np.arange(15).reshape(3,5)
print(arr1)
print(arr1.shape)
print(arr1.dtype)
arr3 = np.zeros((3,4))
print(arr3)
#넘파이 _ 사칙연산
#배열 선언
arr4 = np.array([[1,2,3], [4,5,6]], dtype = np.float64)
arr5 = np.array([[7,8,9], [10,11,12]], dtype = np.float64)
#사칙연산 + - * /
print("arr4 + arr5 = ")
print(arr4 + arr5, "\n")
print("arr4 - arr5 = ")
print(arr4 - arr5, "\n")
print("arr4 * arr5 = ")
print(arr4 * arr5, "\n")
print("arr4 / arr5 = ")
print(arr4 / arr5, "\n")


