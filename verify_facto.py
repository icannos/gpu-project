'''
usage :

$ nvcc LDLt.cu utils.cu -o LDLt
$ ./LDLt | python verify_facto.py --atol 1e-4

'''
import numpy as np
from numpy import nan
from sys import stdin
import argparse
import matplotlib.pyplot as plt




################################################################################
## args parser
################################################################################

parser = argparse.ArgumentParser(description='LDLt factorization and solver tester ')
parser.add_argument("--atol", metavar='A', type=float, default=1e-4, help='the maximum absolute tolerance between values')
args = parser.parse_args()



################################################################################
## parse stdin
################################################################################

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

if not np.all(A>0):
    raise ValueError('matrix A not définie positive !')



################################################################################
# test factorisation :
################################################################################

if D is not None and L is not None:
    D = np.stack([np.diag(m) for m in np.array(D).reshape((n,d))], axis=0)
    L = np.array(L).reshape((n,d,d))
    for i in range(n):
        LDLt = np.linalg.multi_dot([L[i], D[i], L[i].T])
        test = np.allclose(A[i],LDLt, atol=args.atol)
        print(i, '\t A == LDLt: ', test)
        if not test: ## if test diden't pass, display matrices :
            print(
                'L (computed on gpu):\n',L[i],
                '\nD (computed on gpu):\n',D[i],
                '\nLDLt (dot product computed by numpy):\n', LDLt,
                '\nA (randomly initialized matrix):\n', A[i])
            diff = np.abs(A[i]-LDLt)
            print('\nabs(LDLt-A)>tolerance:\n', (diff>args.atol).astype(np.int))
            plt.imshow(diff)
            plt.colorbar()
            plt.title('-- np.abs(A-LDLt) -- matrix '+str(i))
            plt.show()
            pseudo_original = L[i]+L[i].T-2*np.identity(d)+D[i]
            print('\nun-touched values:\n', (pseudo_original==A[i]).astype(np.int))



################################################################################
# test résolution :
################################################################################

if X is not None and Y is not None:
    X = np.array(X).reshape((n,d))
    Y = np.array(Y).reshape((n,d))
    for i in range(n):
        AX = np.linalg.multi_dot([A[i], X[i]])
        test = np.allclose(Y[i],AX, atol=args.atol)
        print(i, '\tAX == Y: ', test)
