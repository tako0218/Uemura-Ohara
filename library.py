import sys
import math
import numpy as numpy
import scipy.sparse.csgraph as ssc
from copy import deepcopy
from fractions import gcd
from itertools import product, accumulate, permutations, combinations
from functools import reduce, lru_cache
from collections import defaultdict, deque, Counter
from heapq import heappop,heappush
from bisect import bisect_left, bisect_right

sys.setrecursionlimit(10 ** 7)

#累乗はpow()

'''
l = [0,1,2,3]
q = deque(l)
q.append(4) # 後ろから4を挿入, l=deque([0,1,2,3,4])
q.appendleft(5)#前から5を挿入, l=deque([5,0,1,2,3,4])
x = q.pop() #後ろの要素を取り出す, x=4, l=deque([5,0,1,2,3])
y = q.popleft() # 前の要素を取り出す, y=5, l = deque([0,1,2,3])

product(l,m)はlとmの直積を返す

A=[1,2,3,4]
for i in permutations(A,2):
    print(i,end=' ')
#(1, 2) (1, 3) (1, 4) (2, 1) (2, 3) (2, 4) (3, 1) (3, 2) (3, 4) (4, 1) (4, 2) (4, 3) 
for i in combinations(A,2):
    print(i, end=' ')
#(1, 2) (1, 3) (1, 4) (2, 3) (2, 4) (3, 4) 

l=['a','b','b','c','b','a','c','c','b','c','b','a']
S=Counter(l)#カウンタークラスが作られる。S=Counter({'b': 5, 'c': 4, 'a': 3})
print(S.most_common(2)) #[('b', 5), ('c', 4)]
print(S.keys()) #dict_keys(['a', 'b', 'c'])
print(S.values()) #dict_values([3, 5, 4])
print(S.items()) #dict_items([('a', 3), ('b', 5), ('c', 4)])

@lru_cache(maxsize=None)
'''
mod = 10 ** 9 + 7

#逆元を用いた高速nCr
U = 2*10**5

fact = [1]*(U+1)
fact_inv = [1]*(U+1)

for i in range(1,U+1):
    fact[i] = (fact[i-1]*i)%mod
fact_inv[U] = pow(fact[U],mod-2,mod)

for i in range(U,0,-1):
    fact_inv[i-1] = (fact_inv[i]*i)%mod

def nCr(n,k,mod):
    if k < 0 or k > n:
        return 0
    x = fact[n]
    x *= fact_inv[k]
    x %= mod
    x *= fact_inv[n-k]
    x %= mod
    return x

def nHr(n,a,mod):
    return nCr(n-1+a,n-1,mod)

#10進数をn進数に変える
def From10_to_n(x,n):
    tmp = x
    to_n = ""
    while tmp > 0:
        to_n = str(tmp%n)+to_n
        tmp = int(tmp/n)
    return to_n

# Union Find-----------------------------------------

# xの根を求める
def find(x):
    if par[x] < 0:
        return x
    else:
        par[x] = find(par[x])
        return par[x]

# xとyの属する集合の併合
def unite(x, y):
    x = find(x)
    y = find(y)

    if x == y:
        return False
    else:
        # sizeの大きいほうがx
        if par[x] > par[y]:
            x, y = y, x
        par[x] += par[y]
        par[y] = x
        return True

# xとyが同じ集合に属するかの判定
def same(x, y):
    return find(x) == find(y)

# xが属する集合の個数
def size(x):
    return -par[find(x)]

# 初期化
# 根なら-size,子なら親の頂点

#nは要素数
'''
n = int(input())
par = [-1] * (4*10**6)
MAX = 0
'''
  # 重み付きUnionFind-----------------------------------
'''
# xの根を求める
def find(x):
    if par[x] < 0:
        return x
    else:
        px = find(par[x])
        wei[x] += wei[par[x]]
        par[x] = px
        return px


# xの根から距離
def weight(x):
    find(x)
    return wei[x]


# w[y]=w[x]+wとなるようにxとyを併合
def unite(x, y, w):
    w += wei[x] - wei[y]
    x = find(x)
    y = find(y)

    if x == y:
        return False
    else:
        # sizeの大きいほうがx
        if par[x] > par[y]:
            x, y = y, x
            w = -w
        par[x] += par[y]
        par[y] = x
        wei[y] = w
        return True


# xとyが同じ集合に属するかの判定
def same(x, y):
    return find(x) == find(y)


# xが属する集合の個数
def size(x):
    return -par[find(x)]


# x,yが同じ集合に属するときのwei[y]-wei[x]
def diff(x, y):
    return weight(y) - weight(x)
'''
#nは要素数
'''
n = int(input())
par = [-1] * n
'''

class BinaryIndexedTree():
    def __init__(self, seq):
        self.size = len(seq)
        self.depth = self.size.bit_length()
        self.build(seq)

    def build(self, seq):
        data = seq
        size = self.size
        for i, x in enumerate(data):
            j = i + (i & (-i))
            if j < size:
                data[j] += data[i]
        self.data = data

    def __repr__(self):
        return self.data.__repr__()

    def get_sum(self, i):
        data = self.data
        s = 0
        while i:
            s += data[i]
            i -= i & -i
        return s

    def add(self, i, x):
        data = self.data
        size = self.size
        while i < size:
            data[i] += x
            i += i & -i

    def find_kth_element(self, k):
        data = self.data
        size = self.size
        x, sx = 0, 0
        dx = 1 << (self.depth)
        for i in range(self.depth - 1, -1, -1):
            dx = (1 << i)
            if x + dx >= size:
                continue
            y = x + dx
            sy = sx + data[y]
            if sy < k:
                x, sx = y, sy
        return x + 1

#(n,0,[[1]])を与えると、n×nの各ベクトルが互いに直交する行列を生成
def make_Hadamard_matrix(n,k,x):
    if n == k:
        return x
    next = [[0] * (2 ** (k+1)) for i in range(2 ** (k+1))]
    for i in range(2 ** (k+1)):
        for j in range(2 ** (k+1)):
            if i >= 2 ** k and j >= 2 ** k:
                next[i][j] = -x[i%(2**k)][j%(2**k)]
            else:
                next[i][j] = x[i%(2**k)][j%(2**k)]
    return make_Hadamard_matrix(n,k+1,next)

def lcm(a,b):
    return a * b // gcd(a,b)