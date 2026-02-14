# Auto-extracted from: P10_v9r6.md
# Attempt: 6

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
math.isprime = sympy.isprime
sympy.crt = crt
sympy.npolycyclotomic = cyclotomic_poly
sympy.n_order = n_order
sympy.multiplicative_order = n_order

def is_power_of_prime(n, p):
    if n <= 0 or p <= 1 or not sympy.isprime(p):
        return False
    # n = p^k ?
    while n % p == 0:
        n //= p
    return n == 1

sympy.isprimepower = is_power_of_prime

def discrete_root(a: int, n: int, m: int, *, all_roots: bool = False):
    """
    Solve x**n ≡ a (mod m).
    Returns:
      - smallest solution if all_roots=False
      - sorted list of all solutions modulo m if all_roots=True

    Notes:
      - Uses sympy.ntheory.residue_ntheory.nthroot_mod for prime/prime-power moduli.
      - For composite m, solves each prime-power modulus then combines via CRT.
    """
    a = int(a); n = int(n); m = int(m)
    if m <= 0 or n <= 0:
        raise ValueError("Require m>0 and n>0")

    a %= m
    if m == 1:
        return [0] if all_roots else 0
    if a == 0:
        # For prime modulus p, only root is 0; for prime powers/composites there can be more,
        # but we keep it simple and return 0 (a valid root).
        return [0] if all_roots else 0

    fac = sp.factorint(m)  # {p: e}
    mod_list = []
    roots_list = []

    for p, e in fac.items():
        pe = p**e
        try:
            r = nthroot_mod(a % pe, n, pe, all_roots=True)  # may return list or None
        except Exception as ex:
            raise NotImplementedError(f"nthroot_mod failed for modulus {pe}: {ex}") from ex
        if not r:
            return [] if all_roots else None
        if not isinstance(r, (list, tuple)):
            r = [r]
        mod_list.append(pe)
        roots_list.append(list(map(int, r)))

    # If only one modulus component, return directly
    if len(mod_list) == 1:
        sols = sorted(set([x % mod_list[0] for x in roots_list[0]]))
        return sols if all_roots else sols[0]

    # Combine all combinations via CRT
    sols = set()
    for combo in product(*roots_list):
        x, mod = crt(mod_list, combo)  # returns (x, lcm_mod) or (None, None)
        if x is not None:
            sols.add(int(x % m))

    sols = sorted(sols)
    return sols if all_roots else (sols[0] if sols else None)

sympy.discrete_root = discrete_root
sympy.ntheory.residue_ntheory.discrete_root = discrete_root

# prevent huge number print error!
sys.set_int_max_str_digits(0)  # unlimited (best-effort)
# avoid recursion stack overflow!
sys.setrecursionlimit(20000)

import pickle
from joblib import Memory
from functools import lru_cache, wraps

memory = Memory(location='aimo3_cache', verbose=0)
_real_cache = memory.cache

def _safe_cache_decorator(func):
    cached_func = _real_cache(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return func(*args, **kwargs)
        return cached_func(*args, **kwargs)
    return wrapper

memory.cache = _safe_cache_decorator

def head(obj, n=5):
    if isinstance(obj, dict):
        return dict(list(obj.items())[:n])
    try:
        return obj[:n]
    except:
        return obj


### Turn 0
import math, itertools
def f_brut(n):
    best=None
    for a in range(1,n-1):
        for b in range(a+1,n):
            c = n - a - b
            if c<=b or c<=0:
                continue
            # a,b,c distinct positive ascending
            # compute lcm
            l = math.lcm(a,b,c)
            if best is None or l<best:
                best=l
                triple=(a,b,c)
    return best, triple
def f_brut2(n):
    # also allow any order distinct, not necessarily sorted.
    best=None
    besttrip=None
    for a in range(1,n-1):
        for b in range(1,n-a):
            if b==a: continue
            c=n-a-b
            if c<=0 or c==a or c==b: continue
            l=math.lcm(a,b,c)
            if best is None or l<best:
                best=l
                besttrip=(a,b,c)
    return best,besttrip
# compute for n up to say 30
res={}
for n in range(6,51):
    best,triple=f_brut2(n)
    res[n]=best
    #print(n,best,triple)
print({n:res[n] for n in range(6,21)})


### Turn 0
import math, itertools

def f_brut(n):
    best = None
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b or c <= 0:
                continue
            l = math.lcm(a, b, c)
            if best is None or l < best:
                best = l
                triple = (a, b, c)
    return (best, triple)

def f_brut2(n):
    best = None
    besttrip = None
    for a in range(1, n - 1):
        for b in range(1, n - a):
            if b == a:
                continue
            c = n - a - b
            if c <= 0 or c == a or c == b:
                continue
            l = math.lcm(a, b, c)
            if best is None or l < best:
                best = l
                besttrip = (a, b, c)
    return (best, besttrip)
res = {}
for n in range(6, 51):
    best, triple = f_brut2(n)
    res[n] = best
print({n: res[n] for n in range(6, 21)})

### Turn 1
def min_lcm(n):
    best,_=f_brut2(n)
    return best
violations=[]
for n in range(6,101):
    best=min_lcm(n)
    if best < (n+1)//2: # D < ceil(n/2)
        violations.append((n,best, (n+1)//2))
violations[:10]


### Turn 1
def min_lcm(n):
    best, _ = f_brut2(n)
    return best
violations = []
for n in range(6, 101):
    best = min_lcm(n)
    if best < (n + 1) // 2:
        violations.append((n, best, (n + 1) // 2))
print(head(violations, 10))

### Turn 2
vals=[]
for n in range(6,101):
    best=min_lcm(n)
    if best == (n+1)//2:
        vals.append(n)
vals[:20], len(vals)


### Turn 2
vals = []
for n in range(6, 101):
    best = min_lcm(n)
    if best == (n + 1) // 2:
        vals.append(n)
print((head(vals, 20), len(vals)))

### Turn 3
vals_even=[]
for n in range(6,101):
    if n%2==0:
        best=min_lcm(n)
        if best == n//2:
            vals_even.append(n)
vals_even[:20], len(vals_even)


### Turn 3
vals_even = []
for n in range(6, 101):
    if n % 2 == 0:
        best = min_lcm(n)
        if best == n // 2:
            vals_even.append(n)
print((head(vals_even, 20), len(vals_even)))

### Turn 4
def f_brut_range(Nmax):
    data=[]
    for n in range(6,Nmax+1):
        data.append((n, min_lcm(n)))
    return data

data = f_brut_range(200)
data[:20]


### Turn 4
def f_brut_range(Nmax):
    data = []
    for n in range(6, Nmax + 1):
        data.append((n, min_lcm(n)))
    return data
data = f_brut_range(200)
print(head(data, 20))

### Turn 5
def compute_f_upto(N):
    f_vals=[0]*(N+1)
    for n in range(6,N+1):
        f_vals[n]=min_lcm(n)
    return f_vals

N=200
f_vals=compute_f_upto(N)
# Show n and f(n), also difference f - ceil(n/2)
diffs=[]
for n in range(6,N+1):
    diff=f_vals[n] - ((n+1)//2)
    diffs.append(diff)
# Let's see distribution
from collections import Counter
cnt=Counter(diffs)
cnt.most_common(10)


### Turn 5
def compute_f_upto(N):
    f_vals = [0] * (N + 1)
    for n in range(6, N + 1):
        f_vals[n] = min_lcm(n)
    return f_vals
N = 200
f_vals = compute_f_upto(N)
diffs = []
for n in range(6, N + 1):
    diff = f_vals[n] - (n + 1) // 2
    diffs.append(diff)
from collections import Counter
cnt = Counter(diffs)
print(cnt.most_common(10))

### Turn 6
for n in range(6,51):
    print(n, f_vals[n], (n+1)//2, f_vals[n]-(n+1)//2)


### Turn 6
for n in range(6, 51):
    print(n, f_vals[n], (n + 1) // 2, f_vals[n] - (n + 1) // 2)

### Turn 7
def best_triple(n):
    best=10**12
    besttr=None
    for a in range(1,n-1):
        for b in range(a+1,n):
            c=n-a-b
            if c<=b or c<=0: continue
            l=math.lcm(a,b,c)
            if l<best:
                best=l
                besttr=(a,b,c)
    return best,besttr

def best_triple_anyorder(n):
    best=10**12
    besttr=None
    for a in range(1,n-1):
        for b in range(1,n-a):
            if b==a: continue
            c=n-a-b
            if c<=0 or c==a or c==b: continue
            l=math.lcm(a,b,c)
            if l<best:
                best=l
                besttr=tuple(sorted((a,b,c)))
    return best,besttr

# test first few n
for n in range(6,21):
    l,tr=best_triple_anyorder(n)
    print(n,l,tr)


### Turn 7
def best_triple(n):
    best = 10 ** 12
    besttr = None
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b or c <= 0:
                continue
            l = math.lcm(a, b, c)
            if l < best:
                best = l
                besttr = (a, b, c)
    return (best, besttr)

def best_triple_anyorder(n):
    best = 10 ** 12
    besttr = None
    for a in range(1, n - 1):
        for b in range(1, n - a):
            if b == a:
                continue
            c = n - a - b
            if c <= 0 or c == a or c == b:
                continue
            l = math.lcm(a, b, c)
            if l < best:
                best = l
                besttr = tuple(sorted((a, b, c)))
    return (best, besttr)
for n in range(6, 21):
    l, tr = best_triple_anyorder(n)
    print(n, l, tr)

### Turn 8
def f_formula(n):
    # n odd >=6
    assert n%2==1
    N_minus1 = n-1
    # check if N-1 is power of 2
    # compute smallest odd divisor >1
    odd_prime = None
    # factor N-1
    import sympy as sp
    fac = sp.factorint(N_minus1)
    # find smallest odd prime divisor >1
    odd_primes = [p for p in fac if p%2==1]
    if odd_primes:
        odd_prime = min(odd_primes)
    if odd_prime is None:
        # N-1 is power of 2
        # We'll need other approach - maybe we try to find minimal D by brute for this small case
        # For now return None
        return None
    # compute d = (N-1)/odd_prime
    d = N_minus1 // odd_prime
    D = n - d - 1
    return D

def test_form(up_to=200):
    mismatches=[]
    for n in range(7, up_to+1,2):
        f_actual,_=f_brut2(n)
        D_form = f_formula(n)
        if D_form is None:
            continue
        if D_form != f_actual:
            mismatches.append((n,f_actual,D_form))
    return mismatches

print(test_form(200)[:20])


### Turn 8
def f_formula(n):
    assert n % 2 == 1
    N_minus1 = n - 1
    odd_prime = None
    import sympy as sp
    fac = sp.factorint(N_minus1)
    odd_primes = [p for p in fac if p % 2 == 1]
    if odd_primes:
        odd_prime = min(odd_primes)
    if odd_prime is None:
        return None
    d = N_minus1 // odd_prime
    D = n - d - 1
    return D

def test_form(up_to=200):
    mismatches = []
    for n in range(7, up_to + 1, 2):
        f_actual, _ = f_brut2(n)
        D_form = f_formula(n)
        if D_form is None:
            continue
        if D_form != f_actual:
            mismatches.append((n, f_actual, D_form))
    return mismatches
print(head(test_form(200), 20))

### Turn 9
for n in range(7,61,2):
    print(n, f_vals[n])


### Turn 9
for n in range(7, 61, 2):
    print(n, f_vals[n])

### Turn 10
ratios=[]
for n in range(7,201,2):
    ratios.append((n, f_vals[n]/n))
ratios[:20]


### Turn 10
ratios = []
for n in range(7, 201, 2):
    ratios.append((n, f_vals[n] / n))
print(head(ratios, 20))

### Turn 11
import sympy as sp
def predict_ratio(n):
    N_minus1 = n-1
    # find smallest odd divisor >1
    odd_div = None
    # iterate over prime divisors
    for p,e in sp.factorint(N_minus1).items():
        if p%2==1:
            if odd_div is None or p<odd_div:
                odd_div=p
    if odd_div is None:
        return None
    return (odd_div-1)/odd_div

for n in range(7,61,2):
    pr = predict_ratio(n)
    act = f_vals[n]/n
    print(n, pr, act)


### Turn 11
import sympy as sp

def predict_ratio(n):
    N_minus1 = n - 1
    odd_div = None
    for p, e in sp.factorint(N_minus1).items():
        if p % 2 == 1:
            if odd_div is None or p < odd_div:
                odd_div = p
    if odd_div is None:
        return None
    return (odd_div - 1) / odd_div
for n in range(7, 61, 2):
    pr = predict_ratio(n)
    act = f_vals[n] / n
    print(n, pr, act)

### Turn 12
ratios = [(n, f_vals[n]/n) for n in range(7,201,2)]
# compute average
avg = sum(r for n,r in ratios)/len(ratios)
avg


### Turn 12
ratios = [(n, f_vals[n] / n) for n in range(7, 201, 2)]
avg = sum((r for n, r in ratios)) / len(ratios)
print(avg)

### Turn 13
print(f_formula(15), f_vals[15])


### Turn 13
print(f_formula(15), f_vals[15])

### Turn 14
def D_form(n):
    N_minus1=n-1
    fac=sp.factorint(N_minus1)
    odd_primes=[p for p in fac if p%2==1]
    if odd_primes:
        q=min(odd_primes)
        D= N_minus1*(q-1)//q
        return D
    else:
        return None

diffs=[]
for n in range(7,201,2):
    Df=D_form(n)
    if Df is None:
        continue
    if Df!=f_vals[n]:
        diffs.append((n,f_vals[n],Df))
len(diffs), diffs[:10]


### Turn 14
def D_form(n):
    N_minus1 = n - 1
    fac = sp.factorint(N_minus1)
    odd_primes = [p for p in fac if p % 2 == 1]
    if odd_primes:
        q = min(odd_primes)
        D = N_minus1 * (q - 1) // q
        return D
    else:
        return None
diffs = []
for n in range(7, 201, 2):
    Df = D_form(n)
    if Df is None:
        continue
    if Df != f_vals[n]:
        diffs.append((n, f_vals[n], Df))
print((len(diffs), head(diffs, 10)))

### Turn 15
odd_f_odd=[]
for n in range(7,201,2):
    if f_vals[n]%2==1:
        odd_f_odd.append((n,f_vals[n]))
odd_f_odd[:20]


### Turn 15
odd_f_odd = []
for n in range(7, 201, 2):
    if f_vals[n] % 2 == 1:
        odd_f_odd.append((n, f_vals[n]))
print(head(odd_f_odd, 20))

### Turn 16
def find_best_triple(n):
    best=10**12
    besttr=None
    for a in range(1,n-1):
        for b in range(a+1,n):
            c=n-a-b
            if c<=b or c<=0: continue
            l=math.lcm(a,b,c)
            if l<best:
                best=l
                besttr=(a,b,c)
    return best,besttr

best, tr = find_best_triple(49)
best, tr


### Turn 16
def find_best_triple(n):
    best = 10 ** 12
    besttr = None
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b or c <= 0:
                continue
            l = math.lcm(a, b, c)
            if l < best:
                best = l
                besttr = (a, b, c)
    return (best, besttr)
best, tr = find_best_triple(49)
print((best, tr))

### Turn 17
import math, sympy as sp
K = math.factorial(2025)
def M_mod(m):
    return pow(3, K, m)

def N_mod_c(c, mod):
    M = M_mod(mod)
    if isinstance(c, int):
        return (M + c) % mod
    else:
        # c may be expression like 4M
        # Not needed now
        pass


### Turn 17
import math, sympy as sp
K = math.factorial(2025)

def M_mod(m):
    return pow(3, K, m)

def N_mod_c(c, mod):
    M = M_mod(mod)
    if isinstance(c, int):
        return (M + c) % mod
    else:
        pass

### Turn 18
import math, sympy as sp, itertools, sys, functools, collections, random, time, math, sys, itertools, hashlib
K = math.factorial(2025)

# candidate pairs (p,q) distinct >1
def candidate_pairs(limit=30):
    pairs=[]
    for p in range(2, limit+1):
        for q in range(p+1, limit+1):
            if p==q: continue
            # distinct
            s = p*q + p + q
            r = (p*q)/s  # ratio as float
            pairs.append((r, p, q, s))
    pairs.sort(key=lambda x: x[0])  # ascending ratio (i.e., smaller D/N)
    return pairs

pairs = candidate_pairs(30)
len(pairs), pairs[:5]


### Turn 18
import math, sympy as sp, itertools, sys, functools, collections, random, time, math, sys, itertools, hashlib
K = math.factorial(2025)

def candidate_pairs(limit=30):
    pairs = []
    for p in range(2, limit + 1):
        for q in range(p + 1, limit + 1):
            if p == q:
                continue
            s = p * q + p + q
            r = p * q / s
            pairs.append((r, p, q, s))
    pairs.sort(key=lambda x: x[0])
    return pairs
pairs = candidate_pairs(30)
print((len(pairs), head(pairs, 5)))

