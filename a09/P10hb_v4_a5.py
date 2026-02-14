# Auto-extracted from: P10hb_v4.md
# Attempt: 5

## stubbed code ##

import math
import numpy
import sympy
import sympy as sp
import itertools
import collections
import mpmath
import sys
import re

from sympy.ntheory.modular import crt
from sympy import cyclotomic_poly
from sympy.ntheory.residue_ntheory import primitive_root, nthroot_mod
from sympy.ntheory import n_order
from itertools import product

mpmath.mp.dps = 64



### Turn 0
import math
from functools import lru_cache

def f_brutal(n,limit=1000):
    # search for smallest N up to limit? Need to be safe.
    # We'll brute up to some max to verify.
    for N in range(1,limit+1):
        divs = [d for d in range(1,N+1) if N%d==0]
        # find any three distinct divisors sum to n
        L=len(divs)
        for i in range(L):
            for j in range(i+1,L):
                for k in range(j+1,L):
                    if divs[i]+divs[j]+divs[k]==n:
                        return N
    return None

def f_using_one(n):
    S=n-1
    min_lcm=10**12
    for a in range(2,S-1):
        b=S-a
        if b<=1 or a==b: continue
        l = (a*b)//math.gcd(a,b)
        if l<min_lcm:
            min_lcm=l
    return min_lcm

def f_exact(n,limit=50000):
    # compute f via checking N up to limit (should be enough for n up to 100)
    for N in range(1,limit+1):
        divs = [d for d in range(1,N+1) if N%d==0]
        # find any three distinct divisors sum to n
        L=len(divs)
        for i in range(L):
            for j in range(i+1,L):
                for k in range(j+1,L):
                    if divs[i]+divs[j]+divs[k]==n:
                        return N
    return None

def compare(Nmax=100):
    mismatches=[]
    for n in range(6,Nmax+1):
        brute = f_brutal(n,limit=10000)
        one = f_using_one(n)
        if brute!=one:
            mismatches.append((n,brute,one))
    return mismatches

print(compare(200)[:20])


### Turn 0
import math
from functools import lru_cache

def f_brutal(n, limit=1000):
    for N in range(1, limit + 1):
        divs = [d for d in range(1, N + 1) if N % d == 0]
        L = len(divs)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
    return None

def f_using_one(n):
    S = n - 1
    min_lcm = 10 ** 12
    for a in range(2, S - 1):
        b = S - a
        if b <= 1 or a == b:
            continue
        l = a * b // math.gcd(a, b)
        if l < min_lcm:
            min_lcm = l
    return min_lcm

def f_exact(n, limit=50000):
    for N in range(1, limit + 1):
        divs = [d for d in range(1, N + 1) if N % d == 0]
        L = len(divs)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
    return None

def compare(Nmax=100):
    mismatches = []
    for n in range(6, Nmax + 1):
        brute = f_brutal(n, limit=10000)
        one = f_using_one(n)
        if brute != one:
            mismatches.append((n, brute, one))
    return mismatches
print(compare(200)[:20])

### Turn 1
def min_lcm_two(S):
    min_l = float('inf')
    best = None
    for a in range(1,S):
        b=S-a
        if a==b: continue
        l = (a*b)//math.gcd(a,b)
        if l < min_l:
            min_l=l
            best=(a,b)
    return min_l,best

def min_lcm_two_small_divisor(S):
    # compute via divisibility heuristic
    # Find smallest divisor d of S with d>=3
    d=None
    for i in range(3,S+1):
        if S%i==0:
            d=i
            break
    if d is None:
        # S prime => d=S
        d=S
    a=S//d
    b=S-a
    return b, (a,b), d

def compare_up_to(N):
    mismatches=[]
    for S in range(4,N+1):
        minl,_=min_lcm_two(S)
        heuristic, pair, d = min_lcm_two_small_divisor(S)
        if minl != heuristic:
            mismatches.append((S,minl,heuristic, d, pair))
    return mismatches

print(compare_up_to(200)[:20])


### Turn 1
def min_lcm_two(S):
    min_l = float('inf')
    best = None
    for a in range(1, S):
        b = S - a
        if a == b:
            continue
        l = a * b // math.gcd(a, b)
        if l < min_l:
            min_l = l
            best = (a, b)
    return (min_l, best)

def min_lcm_two_small_divisor(S):
    d = None
    for i in range(3, S + 1):
        if S % i == 0:
            d = i
            break
    if d is None:
        d = S
    a = S // d
    b = S - a
    return (b, (a, b), d)

def compare_up_to(N):
    mismatches = []
    for S in range(4, N + 1):
        minl, _ = min_lcm_two(S)
        heuristic, pair, d = min_lcm_two_small_divisor(S)
        if minl != heuristic:
            mismatches.append((S, minl, heuristic, d, pair))
    return mismatches
print(compare_up_to(200)[:20])

### Turn 2
def f_one_formula(n):
    S=n-1
    # find smallest prime divisor p of S (p>=2)
    # find smallest p>1 dividing S.
    # Could be 2.
    p=None
    for i in range(2,int(math.sqrt(S))+1):
        if S%i==0:
            p=i
            break
    if p is None:
        p=S # S is prime
    # compute N = S*(p-1)/p
    N = S - S//p
    return N

def compare_f_one(Nmax=200):
    mism=[]
    for n in range(6,Nmax+1):
        brute = f_brutal(n,limit=2000)
        formula = f_one_formula(n)
        if brute!=formula:
            mism.append((n,brute,formula))
    return mism

print(compare_f_one(200)[:30])


### Turn 2
def f_one_formula(n):
    S = n - 1
    p = None
    for i in range(2, int(math.sqrt(S)) + 1):
        if S % i == 0:
            p = i
            break
    if p is None:
        p = S
    N = S - S // p
    return N

def compare_f_one(Nmax=200):
    mism = []
    for n in range(6, Nmax + 1):
        brute = f_brutal(n, limit=2000)
        formula = f_one_formula(n)
        if brute != formula:
            mism.append((n, brute, formula))
    return mism
print(compare_f_one(200)[:30])

### Turn 3
cs=[0,1848374,10162574,265710644,44636594]
mods=[c%6 for c in cs]
mods


### Turn 3
cs = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 6 for c in cs]
print(mods)

### Turn 4
def find_triples(N):
    divs=[d for d in range(1,N+1) if N%d==0]
    triples=[]
    L=len(divs)
    for i in range(L):
        for j in range(i+1,L):
            for k in range(j+1,L):
                if divs[i]+divs[j]+divs[k]==21:
                    triples.append((divs[i],divs[j],divs[k]))
    return triples

print(find_triples(12))


### Turn 4
def find_triples(N):
    divs = [d for d in range(1, N + 1) if N % d == 0]
    triples = []
    L = len(divs)
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                if divs[i] + divs[j] + divs[k] == 21:
                    triples.append((divs[i], divs[j], divs[k]))
    return triples
print(find_triples(12))

### Turn 5
pairs=[]
for p in range(2,21):
    for q in range(2,21):
        if p!=q:
            D=p*q + p + q
            r = p*q/D
            pairs.append((r, p, q, D))
pairs_sorted=sorted(pairs, reverse=True) # larger ratio first (i.e., smaller N)
pairs_sorted[:10]


### Turn 5
pairs = []
for p in range(2, 21):
    for q in range(2, 21):
        if p != q:
            D = p * q + p + q
            r = p * q / D
            pairs.append((r, p, q, D))
pairs_sorted = sorted(pairs, reverse=True)
print(pairs_sorted[:10])

### Turn 6
pairs_sorted_asc=sorted(pairs, key=lambda x: x[0])  # ascending ratio
pairs_sorted_asc[:10]


### Turn 6
pairs_sorted_asc = sorted(pairs, key=lambda x: x[0])
print(pairs_sorted_asc[:10])

### Turn 7
def reduced_denominator(p,q):
    D=p*q + p + q
    g=math.gcd(p*q,D)
    return D//g

pairs_pow3=[]
for p in range(2,31):
    for q in range(2,31):
        if p==q: continue
        k=reduced_denominator(p,q)
        # check if k is power of 3
        tmp=k
        while tmp%3==0 and tmp>1:
            tmp//=3
        if tmp==1:
            pairs_pow3.append((p,q,k))
pairs_pow3[:20], len(pairs_pow3)


### Turn 7
def reduced_denominator(p, q):
    D = p * q + p + q
    g = math.gcd(p * q, D)
    return D // g
pairs_pow3 = []
for p in range(2, 31):
    for q in range(2, 31):
        if p == q:
            continue
        k = reduced_denominator(p, q)
        tmp = k
        while tmp % 3 == 0 and tmp > 1:
            tmp //= 3
        if tmp == 1:
            pairs_pow3.append((p, q, k))
print((pairs_pow3[:20], len(pairs_pow3)))

### Turn 8
def pair_info(p,q):
    D=p*q + p + q
    g=math.gcd(p*q,D)
    k=D//g
    ratio=p*q/D
    return {'p':p,'q':q,'ratio':ratio,'k':k}
pair_info(3,9)


### Turn 8
def pair_info(p, q):
    D = p * q + p + q
    g = math.gcd(p * q, D)
    k = D // g
    ratio = p * q / D
    return {'p': p, 'q': q, 'ratio': ratio, 'k': k}
print(pair_info(3, 9))

### Turn 9
pair_info(3,12)


### Turn 9
print(pair_info(3, 12))

### Turn 10
cs=[0,1848374,10162574,265710644,44636594]
[ c%3 for c in cs]


### Turn 10
cs = [0, 1848374, 10162574, 265710644, 44636594]
print([c % 3 for c in cs])

### Turn 11
def modlist(cs,m):
    return [c % m for c in cs]
modlist(cs,11)


### Turn 11
def modlist(cs, m):
    return [c % m for c in cs]
print(modlist(cs, 11))

### Turn 12
modlist(cs,7)


### Turn 12
print(modlist(cs, 7))

### Turn 13
modlist(cs,17)


### Turn 13
print(modlist(cs, 17))

### Turn 14
modlist(cs,5)


### Turn 14
print(modlist(cs, 5))

### Turn 15
pairs_k5=[]
for p in range(2,51):
    for q in range(2,51):
        if p==q: continue
        k=reduced_denominator(p,q)
        if k==5:
            pairs_k5.append((p,q, p*q/(p*q + p + q)))
pairs_k5[:20], len(pairs_k5)


### Turn 15
pairs_k5 = []
for p in range(2, 51):
    for q in range(2, 51):
        if p == q:
            continue
        k = reduced_denominator(p, q)
        if k == 5:
            pairs_k5.append((p, q, p * q / (p * q + p + q)))
print((pairs_k5[:20], len(pairs_k5)))

### Turn 16
[ c%2 for c in cs]


### Turn 16
print([c % 2 for c in cs])

### Turn 17
modlist(cs,13)


### Turn 17
print(modlist(cs, 13))

### Turn 18
[(c, (1 + c) % 13) for c in cs]


### Turn 18
print([(c, (1 + c) % 13) for c in cs])

### Turn 19
import sympy as sp
def smallest_odd_prime_factor(x):
    if x%2==0:
        # we need odd prime factor
        pass
    # find smallest odd prime divisor >2
    for p in sp.primerange(3, 10000):
        if x % p == 0:
            return p
    return None

for c in cs[1:]:
    print(c, sp.factorint(c))


### Turn 19
import sympy as sp

def smallest_odd_prime_factor(x):
    if x % 2 == 0:
        pass
    for p in sp.primerange(3, 10000):
        if x % p == 0:
            return p
    return None
for c in cs[1:]:
    print(c, sp.factorint(c))

### Turn 20
sp.factorint(5081287)


### Turn 20
print(sp.factorint(5081287))

### Turn 21
sp.isprime(5081287)


### Turn 21
print(sp.isprime(5081287))

### Turn 22
import math, itertools, sys, sympy as sp

# compute factorial
fact_2025 = math.factorial(2025)

def v3(x):
    cnt=0
    while x%3==0 and x>0:
        cnt+=1
        x//=3
    return cnt

def M_mod(mod):
    # compute M = 3^{2025!} mod mod
    if mod == 1:
        return 0
    # factor out powers of 3
    a = 0
    m = mod
    while m % 3 == 0:
        a += 1
        m //= 3
    # m now coprime to 3
    # For m>1, exponent mod phi(m) = 0 because exponent is multiple of phi(m)
    if m == 1:
        # modulus is power of 3 only
        # 3^{2025!} mod 3^a = 0 if exponent >= a
        # Since exponent >= a (as exponent huge) => 0
        return 0
    else:
        # M ≡ 0 (mod 3^a) and ≡ 1 (mod m)
        # combine via CRT
        # use sympy.crt
        mod_list = []
        rem_list = []
        if a>0:
            mod_list.append(3**a)
            rem_list.append(0)
        mod_list.append(m)
        rem_list.append(1)
        x, l = sp.crt(mod_list, rem_list)
        return int(x % mod)
    
def n_mod(c, mod):
    # compute (M + c) mod mod
    return (M_mod(mod) + c % mod) % mod

def candidate_pairs(limit_p=30):
    pairs=[]
    for p in range(2,limit_p+1):
        for q in range(2,limit_p+1):
            if p==q: continue
            D=p*q + p + q
            g=math.gcd(p*q, D)
            denom=D//g
            num=(p*q)//g
            pairs.append((p,q, D, num, denom))
    return pairs

pairs=candidate_pairs(30)

def find_min_lambda(c):
    # returns minimal lambda as Fraction (num/denom) and corresponding (p,q)
    min_frac=None
    best_pair=None
    n_mod_funcs = {}
    # n = M + c
    for (p,q,D,num,denom) in pairs:
        # check condition denom divides n i.e., n % denom ==0
        if n_mod(c, denom)!=0:
            continue
        # also need N divisible by p and q
        # condition D | n*p and D | n*q
        # But if denom divides n, and D = denom * g (where g = gcd(pq,D)), we must check D | n*p and D | n*q
        # Let's just compute directly using modular check:
        if n_mod(c, D) * (p*q) % D != 0:
            # Actually we need n * p*q % D ==0
            # compute n_mod(D) then multiply
            n_mod_D = n_mod(c, D)
            if (n_mod_D * (p*q)) % D != 0:
                continue
        # However we also need N divisible by p and q: N = num * (n // denom)
        # Since denom divides n, n//denom integer.
        # N/p = (num // p) * (n // denom) if p divides num else rational? Wait num = (p*q)/g may not be divisible by p.
        # Simpler: check D divides n * p and D divides n * q
        if n_mod(c, D) * p % D != 0:
            continue
        if n_mod(c, D) * q % D != 0:
            continue
        # If passes, compute lambda = num/denom
        frac = sp.Rational(num, denom)
        if min_frac is None or frac < min_frac:
            min_frac = frac
            best_pair = (p,q)
    return min_frac, best_pair

# test for c values
c_values=[0,1848374,10162574,265710644,44636594]
for c in c_values:
    lam, pair = find_min_lambda(c)
    print(c, lam, pair)


### Turn 22
import math, itertools, sys, sympy as sp
fact_2025 = math.factorial(2025)

def v3(x):
    cnt = 0
    while x % 3 == 0 and x > 0:
        cnt += 1
        x //= 3
    return cnt

def M_mod(mod):
    if mod == 1:
        return 0
    a = 0
    m = mod
    while m % 3 == 0:
        a += 1
        m //= 3
    if m == 1:
        return 0
    else:
        mod_list = []
        rem_list = []
        if a > 0:
            mod_list.append(3 ** a)
            rem_list.append(0)
        mod_list.append(m)
        rem_list.append(1)
        x, l = sp.crt(mod_list, rem_list)
        return int(x % mod)

def n_mod(c, mod):
    return (M_mod(mod) + c % mod) % mod

def candidate_pairs(limit_p=30):
    pairs = []
    for p in range(2, limit_p + 1):
        for q in range(2, limit_p + 1):
            if p == q:
                continue
            D = p * q + p + q
            g = math.gcd(p * q, D)
            denom = D // g
            num = p * q // g
            pairs.append((p, q, D, num, denom))
    return pairs
pairs = candidate_pairs(30)

def find_min_lambda(c):
    min_frac = None
    best_pair = None
    n_mod_funcs = {}
    for p, q, D, num, denom in pairs:
        if n_mod(c, denom) != 0:
            continue
        if n_mod(c, D) * (p * q) % D != 0:
            n_mod_D = n_mod(c, D)
            if n_mod_D * (p * q) % D != 0:
                continue
        if n_mod(c, D) * p % D != 0:
            continue
        if n_mod(c, D) * q % D != 0:
            continue
        frac = sp.Rational(num, denom)
        if min_frac is None or frac < min_frac:
            min_frac = frac
            best_pair = (p, q)
    return (min_frac, best_pair)
c_values = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_values:
    lam, pair = find_min_lambda(c)
    print(c, lam, pair)

### Turn 23
from sympy.ntheory.modular import crt
def M_mod(mod):
    if mod == 1:
        return 0
    a = 0
    m = mod
    while m % 3 == 0:
        a += 1
        m //= 3
    if m == 1:
        return 0
    else:
        # combine x ≡ 0 mod 3^a (if a>0) and x ≡ 1 mod m
        if a > 0:
            x, l = crt([3**a, m], [0, 1])
        else:
            x, l = crt([m], [1])
        return int(x % mod)
    
def n_mod(c, mod):
    return (M_mod(mod) + c % mod) % mod

# Test M_mod for some values
for mod in [1,2,3,5,7,11,13,27]:
    print(mod, M_mod(mod))


### Turn 23
from sympy.ntheory.modular import crt

def M_mod(mod):
    if mod == 1:
        return 0
    a = 0
    m = mod
    while m % 3 == 0:
        a += 1
        m //= 3
    if m == 1:
        return 0
    else:
        if a > 0:
            x, l = crt([3 ** a, m], [0, 1])
        else:
            x, l = crt([m], [1])
        return int(x % mod)

def n_mod(c, mod):
    return (M_mod(mod) + c % mod) % mod
for mod in [1, 2, 3, 5, 7, 11, 13, 27]:
    print(mod, M_mod(mod))

### Turn 24
def find_min_lambda(c, max_p=30):
    min_frac = None
    best_pair = None
    for p in range(2, max_p+1):
        for q in range(2, max_p+1):
            if p==q: continue
            D = p*q + p + q
            # condition D divides n*p and n*q
            n_mod_D = n_mod(c, D)
            if (n_mod_D * p) % D != 0:
                continue
            if (n_mod_D * q) % D != 0:
                continue
            # now compute fraction lambda = p*q/D reduced
            g = math.gcd(p*q, D)
            num = (p*q)//g
            denom = D//g
            # ensure denom divides n (since after reduction, need integer)
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p,q)
    return min_frac, best_pair

c_vals=[0,1848374,10162574,265710644,44636594]
for c in c_vals:
    lam,pair=find_min_lambda(c, max_p=30)
    print(c, lam, pair)


### Turn 24
def find_min_lambda(c, max_p=30):
    min_frac = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(2, max_p + 1):
            if p == q:
                continue
            D = p * q + p + q
            n_mod_D = n_mod(c, D)
            if n_mod_D * p % D != 0:
                continue
            if n_mod_D * q % D != 0:
                continue
            g = math.gcd(p * q, D)
            num = p * q // g
            denom = D // g
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p, q)
    return (min_frac, best_pair)
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    lam, pair = find_min_lambda(c, max_p=30)
    print(c, lam, pair)

### Turn 25
def find_min_lambda(c, max_p=100):
    min_frac = None
    best_pair = None
    for p in range(2, max_p+1):
        for q in range(2, max_p+1):
            if p==q: continue
            D = p*q + p + q
            n_mod_D = n_mod(c, D)
            if (n_mod_D * p) % D != 0:
                continue
            if (n_mod_D * q) % D != 0:
                continue
            # compute reduced fraction
            g = math.gcd(p*q, D)
            num = (p*q)//g
            denom = D//g
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p,q)
    return min_frac, best_pair

for c in c_vals[2:]:
    lam, pair = find_min_lambda(c, max_p=100)
    print(c, lam, pair)


### Turn 25
def find_min_lambda(c, max_p=100):
    min_frac = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(2, max_p + 1):
            if p == q:
                continue
            D = p * q + p + q
            n_mod_D = n_mod(c, D)
            if n_mod_D * p % D != 0:
                continue
            if n_mod_D * q % D != 0:
                continue
            g = math.gcd(p * q, D)
            num = p * q // g
            denom = D // g
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p, q)
    return (min_frac, best_pair)
for c in c_vals[2:]:
    lam, pair = find_min_lambda(c, max_p=100)
    print(c, lam, pair)

### Turn 26
for c in c_vals[1:]:
    lam, pair = find_min_lambda(c, max_p=200)
    print(c, lam, pair)


### Turn 26
for c in c_vals[1:]:
    lam, pair = find_min_lambda(c, max_p=200)
    print(c, lam, pair)

### Turn 27
def n_mod_aM(a, mod):
    # n = a * M
    return (a * M_mod(mod)) % mod

def find_min_lambda_aM(a, max_p=30):
    min_frac=None
    best_pair=None
    for p in range(2, max_p+1):
        for q in range(2, max_p+1):
            if p==q: continue
            D = p*q + p + q
            n_mod_D = n_mod_aM(a, D)
            if (n_mod_D * p) % D != 0:
                continue
            if (n_mod_D * q) % D != 0:
                continue
            g=math.gcd(p*q, D)
            num=(p*q)//g
            denom=D//g
            # need D|n*p and D|n*q already checked; need denom|n
            # n = a*M, we need n_mod(denom) == 0
            if n_mod_aM(a, denom)!=0:
                continue
            frac=sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac=frac
                best_pair=(p,q)
    return min_frac, best_pair

lam_a5, pair_a5 = find_min_lambda_aM(5, max_p=30)
lam_a5, pair_a5


### Turn 27
def n_mod_aM(a, mod):
    return a * M_mod(mod) % mod

def find_min_lambda_aM(a, max_p=30):
    min_frac = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(2, max_p + 1):
            if p == q:
                continue
            D = p * q + p + q
            n_mod_D = n_mod_aM(a, D)
            if n_mod_D * p % D != 0:
                continue
            if n_mod_D * q % D != 0:
                continue
            g = math.gcd(p * q, D)
            num = p * q // g
            denom = D // g
            if n_mod_aM(a, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p, q)
    return (min_frac, best_pair)
lam_a5, pair_a5 = find_min_lambda_aM(5, max_p=30)
print((lam_a5, pair_a5))

### Turn 28
from fractions import Fraction
S = Fraction(2,3)+Fraction(2,3)+Fraction(16,25)+Fraction(30,47)+Fraction(64,97)+Fraction(110,167)
S


### Turn 28
from fractions import Fraction
S = Fraction(2, 3) + Fraction(2, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print(S)

### Turn 29
p,q=S.numerator, S.denominator
mod = 99991
(p+q) % mod


### Turn 29
p, q = (S.numerator, S.denominator)
mod = 99991
print((p + q) % mod)

### Turn 30
c=10162574
c%47


### Turn 30
c = 10162574
print(c % 47)

### Turn 31
def find_min_lambda_big(c, max_p=30, max_q=500):
    min_frac=None
    best_pair=None
    for p in range(2, max_p+1):
        for q in range(2, max_q+1):
            if p==q: continue
            D = p*q + p + q
            n_mod_D = n_mod(c, D)
            if (n_mod_D * p) % D != 0: continue
            if (n_mod_D * q) % D != 0: continue
            g=math.gcd(p*q, D)
            num=(p*q)//g
            denom=D//g
            if n_mod(c, denom)!=0:
                continue
            frac=sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac=frac
                best_pair=(p,q)
    return min_frac,best_pair

for c in c_vals[1:]:
    lam,pair=find_min_lambda_big(c, max_p=30, max_q=500)
    print(c, lam, pair)


### Turn 31
def find_min_lambda_big(c, max_p=30, max_q=500):
    min_frac = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(2, max_q + 1):
            if p == q:
                continue
            D = p * q + p + q
            n_mod_D = n_mod(c, D)
            if n_mod_D * p % D != 0:
                continue
            if n_mod_D * q % D != 0:
                continue
            g = math.gcd(p * q, D)
            num = p * q // g
            denom = D // g
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p, q)
    return (min_frac, best_pair)
for c in c_vals[1:]:
    lam, pair = find_min_lambda_big(c, max_p=30, max_q=500)
    print(c, lam, pair)

### Turn 32
def find_min_lambda_full(c, max_val=200):
    min_frac=None
    best_pair=None
    for p in range(2, max_val+1):
        for q in range(2, max_val+1):
            if p==q: continue
            D = p*q + p + q
            n_mod_D = n_mod(c, D)
            if (n_mod_D * p) % D != 0: continue
            if (n_mod_D * q) % D != 0: continue
            g=math.gcd(p*q, D)
            num=(p*q)//g
            denom=D//g
            if n_mod(c, denom)!=0:
                continue
            frac=sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac=frac
                best_pair=(p,q)
    return min_frac,best_pair

for c in c_vals[1:]:
    lam,pair=find_min_lambda_full(c, max_val=200)
    print(c, lam, pair)


### Turn 32
def find_min_lambda_full(c, max_val=200):
    min_frac = None
    best_pair = None
    for p in range(2, max_val + 1):
        for q in range(2, max_val + 1):
            if p == q:
                continue
            D = p * q + p + q
            n_mod_D = n_mod(c, D)
            if n_mod_D * p % D != 0:
                continue
            if n_mod_D * q % D != 0:
                continue
            g = math.gcd(p * q, D)
            num = p * q // g
            denom = D // g
            if n_mod(c, denom) != 0:
                continue
            frac = sp.Rational(num, denom)
            if min_frac is None or frac < min_frac:
                min_frac = frac
                best_pair = (p, q)
    return (min_frac, best_pair)
for c in c_vals[1:]:
    lam, pair = find_min_lambda_full(c, max_val=200)
    print(c, lam, pair)

### Turn 33
from fractions import Fraction
lambdas = [Fraction(2,3), Fraction(2,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)]
S = sum(lambdas, Fraction(0,1))
S_numer=S.numerator
S_denom=S.denominator
S_numer, S_denom, S_numer+S_denom, (S_numer+S_denom)%99991


### Turn 33
from fractions import Fraction
lambdas = [Fraction(2, 3), Fraction(2, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)]
S = sum(lambdas, Fraction(0, 1))
S_numer = S.numerator
S_denom = S.denominator
print((S_numer, S_denom, S_numer + S_denom, (S_numer + S_denom) % 99991))

