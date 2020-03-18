'''
usage :

$ nvcc LDLt.cu utils.cu -o LDLt
$ ./LDLt | python test2.py --atol 1e-4

'''
import numpy as np
from numpy import nan
from sys import stdin
import argparse

parser = argparse.ArgumentParser(description='LDLt factorization and solver tester ')
parser.add_argument("--atol", metavar='A', type=float, default=1e-4, help='the maximum absolute tolerance between values')


args = parser.parse_args()



dict_str = ''
start = False
for l in stdin.readlines():
    if start or l[0]=='{':
        start = True
        dict_str += l

dict_val = eval(dict_str)

A = dict_val.get('A', None)
D = dict_val.get('D', None)
L = dict_val.get('L', None)
Y = dict_val.get('Y', None)
X = dict_val.get('X', None)


print('found : ', dict_val.keys())

if A is None:
    raise KeyError('A')
A = np.array(A)

if len(A.shape)==3:
    n,d,d2 = A.shape
elif len(A.shape)==2:
    d,d2 = A.shape
    n = 1
else:
    raise ValueError
assert d==d2
print('n={}, d={}'.format(n,d))
A = A.reshape((n,d,d))

# factorisation :
if D is not None and L is not None:
    D = np.diag(D).reshape((n,d,d))
    L = np.array(L).reshape((n,d,d))
    for i in range(n):
        LDLt = np.linalg.multi_dot([L[i], D[i], L[i].T])
        test = np.allclose(A[i],LDLt, atol=args.atol)
        print(i, ' A == LDLt: ', test)
        if not test:
            print(L[i], D[i], LDLt, A[i])

if X is not None and Y is not None:
    X = np.array(X).reshape((n,d))
    Y = np.array(Y).reshape((n,d))
    for i in range(n):
        AX = np.linalg.multi_dot([A[i], X[i]])
        test = np.allclose(Y[i],AX, atol=args.atol)
        print(i, '\tAX == Y: ', test)
