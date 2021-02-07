import sys
import math
import itertools
from functools import reduce
from collections import defaultdict

mod = 10 ** 9 + 7
  # 累乗はpow()
def nCr(N,a):
    num = reduce(lambda x,y:x * y % mod,range(N,N-a,-1))
    den = reduce(lambda x,y:x * y % mod,range(1,a+1))
    return num * pow(den,mod-2,mod)%mod

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