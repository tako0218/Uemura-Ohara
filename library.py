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