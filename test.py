import numpy as np
from sys import stdin

fromc = list(stdin.readlines())


L = np.array(eval(fromc[0]))
Lt = np.transpose(L)

D = np.array(eval(fromc[1]))
Y = np.array(eval(fromc[2]))
Xchap = np.array(eval(fromc[3]))

print("L:\n", L)
print("Lt:\n", Lt)
print("D:", D)
print("Y:", Y)

print(f"True X: {np.dot(np.linalg.inv(np.dot(L, Lt)), Y)}")
print(f"X: {Xchap}")