from functools import reduce
import itertools
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