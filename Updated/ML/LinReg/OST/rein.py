def hcf(n,m):
    if n==0 or m==0:
        return n+m
    if n==m:
        return n
    if n>m:
        hcf(n-m,m)
    else:
        hcf(n,m-n)

print(hcf(4,12))