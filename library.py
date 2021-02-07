from functools import reduce
mod = 10 ** 9 + 7
  # 累乗はpow()
def nCr(N,a):
    num = reduce(lambda x,y:x * y % mod,range(N,N-a,-1))
    den = reduce(lambda x,y:x * y % mod,range(1,a+1))
    return num * pow(den,mod-2,mod)%mod