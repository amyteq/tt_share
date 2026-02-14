# Auto-extracted from: P10_v9r6.md
# Attempt: 3

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
import sympy as sp

def f_of_n(n):
    # find smallest positive integer k with three distinct divisors adding to n
    # search from 1 upwards
    k=1
    while True:
        # compute divisors of k
        divs = sp.divisors(k)
        if len(divs)>=3:
            # need distinct triple sum to n
            # we can brute
            dlist = divs
            # triple combos
            for i in range(len(dlist)):
                for j in range(i+1,len(dlist)):
                    for kidx in range(j+1,len(dlist)):
                        if dlist[i]+dlist[j]+dlist[kidx]==n:
                            return k
        k+=1

# test for n small values
vals = {}
for n in range(6,30):
    vals[n]=f_of_n(n)
vals


### Turn 0
import sympy as sp

def f_of_n(n):
    k = 1
    while True:
        divs = sp.divisors(k)
        if len(divs) >= 3:
            dlist = divs
            for i in range(len(dlist)):
                for j in range(i + 1, len(dlist)):
                    for kidx in range(j + 1, len(dlist)):
                        if dlist[i] + dlist[j] + dlist[kidx] == n:
                            return k
        k += 1
vals = {}
for n in range(6, 30):
    vals[n] = f_of_n(n)
print(vals)

### Turn 1
vals = {}
for n in range(6,101):
    vals[n]=f_of_n(n)
vals[:10]


### Turn 1
vals = {}
for n in range(6, 101):
    vals[n] = f_of_n(n)
print(head(vals, 10))

### Turn 2
list(vals.items())[:20]


### Turn 2
print(head(list(vals.items()), 20))

### Turn 3
list(vals.items())[20:40]


### Turn 3
print(head(list(vals.items()), 40))

### Turn 4
ratios = {n: sp.Rational(vals[n], n) for n in vals}
sorted(set(ratios.values()))[:20]


### Turn 4
ratios = {n: sp.Rational(vals[n], n) for n in vals}
print(head(sorted(set(ratios.values())), 20))

### Turn 5
def f_by_lcm(n):
    s=n-1
    best=None
    for a in range(2, s-1):
        b=s-a
        if b<=1 or a==b:
            continue
        l=sp.ilcm(a,b)
        if best is None or l<best:
            best=l
    return best

vals2={}
for n in range(6,100):
    vals2[n]=f_by_lcm(n)
diff=[n for n in range(6,100) if vals[n]!=vals2[n]]
diff[:20], len(diff)


### Turn 5
def f_by_lcm(n):
    s = n - 1
    best = None
    for a in range(2, s - 1):
        b = s - a
        if b <= 1 or a == b:
            continue
        l = sp.ilcm(a, b)
        if best is None or l < best:
            best = l
    return best
vals2 = {}
for n in range(6, 100):
    vals2[n] = f_by_lcm(n)
diff = [n for n in range(6, 100) if vals[n] != vals2[n]]
print((head(diff, 20), len(diff)))

### Turn 6
def f_brutal(n):
    best=None
    # enumerate distinct triples a<b<c (since ordering irrelevant)
    for a in range(1, n-2):
        for b in range(a+1, n-a-1):
            c=n-a-b
            if c<=b: # ensure distinct and ascending
                continue
            L=sp.ilcm(a,b,c)
            if best is None or L<best:
                best=L
    return best

vals_br={}
for n in range(6,50):
    vals_br[n]=f_brutal(n)
diff=[n for n in range(6,50) if vals[n]!=vals_br[n]]
diff[:20], len(diff)


### Turn 6
def f_brutal(n):
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - a - 1):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if best is None or L < best:
                best = L
    return best
vals_br = {}
for n in range(6, 50):
    vals_br[n] = f_brutal(n)
diff = [n for n in range(6, 50) if vals[n] != vals_br[n]]
print((head(diff, 20), len(diff)))

### Turn 7
def f_brutal_limited(n):
    # use previously defined function for small n
    return f_brutal(n)
def compute_f_for_numbers(nums):
    res={}
    for n in nums:
        res[n]=f_brutal_limited(n)
    return res

powers=[3**i for i in range(1,9)]
compute_f_for_numbers(powers)


### Turn 7
def f_brutal_limited(n):
    return f_brutal(n)

def compute_f_for_numbers(nums):
    res = {}
    for n in nums:
        res[n] = f_brutal_limited(n)
    return res
powers = [3 ** i for i in range(1, 9)]
print(compute_f_for_numbers(powers))

### Turn 8
import sympy as sp, math, sys, itertools, collections, math, random, fractions, decimal, json, pprint, typing, fractions, math, itertools, functools, collections, math, sys, os, types, inspect, math, math
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c))


### Turn 8
import sympy as sp, math, sys, itertools, collections, math, random, fractions, decimal, json, pprint, typing, fractions, math, itertools, functools, collections, math, sys, os, types, inspect, math, math
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c))

### Turn 9
def f_brutal_min(n):
    return f_brutal(n)

def ratio_f_to_n3(n):
    return f_brutal_min(n) - math.ceil(n/3)

diffs={}
for n in range(6,201):
    dif = f_brutal_min(n) - math.ceil(n/3)
    diffs[n]=dif
# find where diff>0
max_diff = max(diffs.values())
max_diff, [n for n in diffs if diffs[n]==max_diff][:10]


### Turn 9
def f_brutal_min(n):
    return f_brutal(n)

def ratio_f_to_n3(n):
    return f_brutal_min(n) - math.ceil(n / 3)
diffs = {}
for n in range(6, 201):
    dif = f_brutal_min(n) - math.ceil(n / 3)
    diffs[n] = dif
max_diff = max(diffs.values())
print((max_diff, head([n for n in diffs if diffs[n] == max_diff], 10)))

### Turn 10
n=199
f_val=f_brutal_min(n)
f_val, math.ceil(n/3), f_val-math.ceil(n/3)


### Turn 10
n = 199
f_val = f_brutal_min(n)
print((f_val, math.ceil(n / 3), f_val - math.ceil(n / 3)))

### Turn 11
def f_min_brutal(n):
    return f_brutal_min(n)
def check_nine(limit):
    res=[]
    for n in range(6,limit+1):
        if n%9==0:
            if f_min_brutal(n) != 2*n//3:
                res.append((n,f_min_brutal(n),2*n//3))
    return res[:20]
check_nine(200)


### Turn 11
def f_min_brutal(n):
    return f_brutal_min(n)

def check_nine(limit):
    res = []
    for n in range(6, limit + 1):
        if n % 9 == 0:
            if f_min_brutal(n) != 2 * n // 3:
                res.append((n, f_min_brutal(n), 2 * n // 3))
    return head(res, 20)
print(check_nine(200))

### Turn 12
def min_triple_lcm_sum(n):
    best=None
    best_trip=None
    for a in range(1,n-1):
        for b in range(a+1,n-a):
            c=n-a-b
            if c<=b:
                continue
            L=sp.ilcm(a,b,c)
            if best is None or L<best:
                best=L; best_trip=(a,b,c)
    return best,best_trip

fval,trip=min_triple_lcm_sum(63)
fval,trip


### Turn 12
def min_triple_lcm_sum(n):
    best = None
    best_trip = None
    for a in range(1, n - 1):
        for b in range(a + 1, n - a):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if best is None or L < best:
                best = L
                best_trip = (a, b, c)
    return (best, best_trip)
fval, trip = min_triple_lcm_sum(63)
print((fval, trip))

### Turn 13
def f_brutal_limited(n):
    return f_brutal_min(n)
# compute for k upto 8
vals={}
for k in range(0,9):
    n=5*3**k
    val=f_brutal_limited(n)
    vals[k]=(n,val)
    print(k, n, val, sp.Rational(val,n))


### Turn 13
def f_brutal_limited(n):
    return f_brutal_min(n)
vals = {}
for k in range(0, 9):
    n = 5 * 3 ** k
    val = f_brutal_limited(n)
    vals[k] = (n, val)
    print(k, n, val, sp.Rational(val, n))

### Turn 14
for k in range(1,9):
    n=5*3**k
    val=f_brutal_min(n)
    print(k,n,val, sp.Rational(val,n))


### Turn 14
for k in range(1, 9):
    n = 5 * 3 ** k
    val = f_brutal_min(n)
    print(k, n, val, sp.Rational(val, n))

### Turn 15
def mod7(k):
    return (5*pow(3,k,7))%7
[mod7(k) for k in range(0,10)]


### Turn 15
def mod7(k):
    return 5 * pow(3, k, 7) % 7
print([mod7(k) for k in range(0, 10)])

### Turn 16
def f_brutal_limit(n):
    # naive triple enumeration, may be slower for 135 but okay
    best=n+1
    best_trip=None
    for a in range(1, n//3):
        for b in range(a+1, (n-a)//2):
            c=n-a-b
            if c<=b:
                continue
            L=sp.ilcm(a,b,c)
            if L<best:
                best=L
                best_trip=(a,b,c)
    return best, best_trip

f135, trip135 = f_brutal_limit(135)
f135, trip135


### Turn 16
def f_brutal_limit(n):
    best = n + 1
    best_trip = None
    for a in range(1, n // 3):
        for b in range(a + 1, (n - a) // 2):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if L < best:
                best = L
                best_trip = (a, b, c)
    return (best, best_trip)
f135, trip135 = f_brutal_limit(135)
print((f135, trip135))

### Turn 17
def f_small(n):
    return f_brutal_limit(n)[0]

def compute_f_seq(t_max):
    vals={}
    for t in range(1,t_max+1):
        n=5*3**t
        vals[t]=f_small(n)
    return vals

vals=compute_f_seq(7)
vals


### Turn 17
def f_small(n):
    return f_brutal_limit(n)[0]

def compute_f_seq(t_max):
    vals = {}
    for t in range(1, t_max + 1):
        n = 5 * 3 ** t
        vals[t] = f_small(n)
    return vals
vals = compute_f_seq(7)
print(vals)

### Turn 18
c = 1848374
c%3


### Turn 18
c = 1848374
print(c % 3)

### Turn 19
def f_fast_brutal(n):
    # use same brute but maybe early exit?
    best=n+1
    for a in range(1, n//3+1):
        for b in range(a+1, (n-a)//2+1):
            c=n-a-b
            if c<=b: continue
            L=sp.ilcm(a,b,c)
            if L<best:
                best=L
    return best

def f_small_range(limit):
    result={}
    for n in range(6,limit+1):
        result[n]=f_fast_brutal(n)
    return result

vals_small = f_small_range(200)
for n in range(90,151):
    if n%3!=0:
        print(n, vals_small[n], vals_small[n]/n)


### Turn 19
def f_fast_brutal(n):
    best = n + 1
    for a in range(1, n // 3 + 1):
        for b in range(a + 1, (n - a) // 2 + 1):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if L < best:
                best = L
    return best

def f_small_range(limit):
    result = {}
    for n in range(6, limit + 1):
        result[n] = f_fast_brutal(n)
    return result
vals_small = f_small_range(200)
for n in range(90, 151):
    if n % 3 != 0:
        print(n, vals_small[n], vals_small[n] / n)

### Turn 20
def min_ratio(limit):
    best_ratio = sp.Rational(1,1)  # start >1 maybe
    best_trip = None
    for a in range(1, limit+1):
        for b in range(a+1, limit+1):
            for c in range(b+1, limit+1):
                L=sp.ilcm(a,b,c)
                s=a+b+c
                ratio = sp.Rational(L, s)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_trip = (a,b,c)
    return best_ratio, best_trip

best_ratio, best_trip = min_ratio(100)
best_ratio, best_trip


### Turn 20
def min_ratio(limit):
    best_ratio = sp.Rational(1, 1)
    best_trip = None
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):
            for c in range(b + 1, limit + 1):
                L = sp.ilcm(a, b, c)
                s = a + b + c
                ratio = sp.Rational(L, s)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_trip = (a, b, c)
    return (best_ratio, best_trip)
best_ratio, best_trip = min_ratio(100)
print((best_ratio, best_trip))

### Turn 21
best_ratio, best_trip = min_ratio(200)
best_ratio, best_trip


### Turn 21
best_ratio, best_trip = min_ratio(200)
print((best_ratio, best_trip))

### Turn 22
c1=1848374
c1%11


### Turn 22
c1 = 1848374
print(c1 % 11)

### Turn 23
c2=10162574
c2%11


### Turn 23
c2 = 10162574
print(c2 % 11)

### Turn 24
c3=265710644
c3%11


### Turn 24
c3 = 265710644
print(c3 % 11)

### Turn 25
c4=44636594
c4%11


### Turn 25
c4 = 44636594
print(c4 % 11)

### Turn 26
cs=[0,1848374,10162574,265710644,44636594]
for c in cs:
    val=c+1
    print(c,val, sp.factorint(val))


### Turn 26
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    val = c + 1
    print(c, val, sp.factorint(val))

### Turn 27
def n_mod(p,c):
    M_mod = pow(3, math.factorial(2025), p) if p!=3 else pow(3, math.factorial(2025), p) # but exponent huge; compute using pow with exponent modulo phi


### Turn 27
def n_mod(p, c):
    M_mod = pow(3, math.factorial(2025), p) if p != 3 else pow(3, math.factorial(2025), p)

### Turn 28
from fractions import Fraction
def ratio(p,q):
    return Fraction(1, 1+Fraction(1,p)+Fraction(1,q))
ratio(5,31)


### Turn 28
from fractions import Fraction

def ratio(p, q):
    return Fraction(1, 1 + Fraction(1, p) + Fraction(1, q))
print(ratio(5, 31))

### Turn 29
cs=[0,1848374,10162574,265710644,44636594]
for c in cs:
    print(c, (c+1)%11)


### Turn 29
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, (c + 1) % 11)

### Turn 30
def compute_min_ratio(limit):
    min_ratio = None
    best = None
    for s in range(6, limit+1):
        f_val = f_brutal_limited(s)
        ratio = sp.Rational(f_val, s)
        if min_ratio is None or ratio < min_ratio:
            min_ratio = ratio
            best = (s, f_val, ratio)
    return best

# compute decreasing order of minimal ratios?
def best_ratios(limit, top=10):
    ratios = []
    for s in range(6, limit+1):
        f_val = f_brutal_limited(s)
        ratios.append((sp.Rational(f_val, s), s, f_val))
    ratios.sort()
    return ratios[:top]

best_ratios(200, 20)[:10]


### Turn 30
def compute_min_ratio(limit):
    min_ratio = None
    best = None
    for s in range(6, limit + 1):
        f_val = f_brutal_limited(s)
        ratio = sp.Rational(f_val, s)
        if min_ratio is None or ratio < min_ratio:
            min_ratio = ratio
            best = (s, f_val, ratio)
    return best

def best_ratios(limit, top=10):
    ratios = []
    for s in range(6, limit + 1):
        f_val = f_brutal_limited(s)
        ratios.append((sp.Rational(f_val, s), s, f_val))
    ratios.sort()
    return head(ratios, top)
print(head(best_ratios(200, 20), 10))

### Turn 31
def distinct_ratios(limit):
    ratios=set()
    for s in range(6, limit+1):
        ratios.add(sp.Rational(f_brutal_limited(s), s))
    return sorted(ratios)

ratios=distinct_ratios(200)
ratios[:20]


### Turn 31
def distinct_ratios(limit):
    ratios = set()
    for s in range(6, limit + 1):
        ratios.add(sp.Rational(f_brutal_limited(s), s))
    return sorted(ratios)
ratios = distinct_ratios(200)
print(head(ratios, 20))

### Turn 32
[(s,f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6,200) if sp.Rational(f_brutal_limited(s), s)==sp.Rational(4,7)]


### Turn 32
print([(s, f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6, 200) if sp.Rational(f_brutal_limited(s), s) == sp.Rational(4, 7)])

### Turn 33
cs_all=[0,1848374,10162574,265710644,44636594]
for c in cs_all:
    print(c, (c+1)%7)


### Turn 33
cs_all = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs_all:
    print(c, (c + 1) % 7)

### Turn 34
[(s,f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6,500) if sp.Rational(f_brutal_limited(s), s)==sp.Rational(10,17)]


### Turn 34
print([(s, f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6, 500) if sp.Rational(f_brutal_limited(s), s) == sp.Rational(10, 17)])

### Turn 35
def best_trip_sum(s):
    best = None
    bestt=None
    for a in range(1,s):
        for b in range(a+1,s-a):
            c=s-a-b
            if c<=b: continue
            L=sp.ilcm(a,b,c)
            if best is None or L<best:
                best=L; bestt=(a,b,c)
    return best, bestt
best_trip_sum(17)


### Turn 35
def best_trip_sum(s):
    best = None
    bestt = None
    for a in range(1, s):
        for b in range(a + 1, s - a):
            c = s - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if best is None or L < best:
                best = L
                bestt = (a, b, c)
    return (best, bestt)
print(best_trip_sum(17))

### Turn 36
for c in cs_all:
    print(c, (c+1) % 17)


### Turn 36
for c in cs_all:
    print(c, (c + 1) % 17)

### Turn 37
[(s,f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6,200) if sp.Rational(f_brutal_limited(s), s)==sp.Rational(3,5)]


### Turn 37
print([(s, f_brutal_limited(s), sp.Rational(f_brutal_limited(s), s)) for s in range(6, 200) if sp.Rational(f_brutal_limited(s), s) == sp.Rational(3, 5)])

### Turn 38
def primitive_ratio(s):
    # s must be such that f(s)/s = minimal among its divisors? We just compute f(s)/s directly, but note that if s has a divisor d with better ratio, then f(s)/s = best ratio (since we can factor). So we just compute f(s)/s using the recurrence as before.
    return sp.Rational(f_brutal_limited(s), s)

ratios = {}
for s in range(6,200):
    ratios[s] = primitive_ratio(s)
sorted_ratios = sorted([(ratios[s], s) for s in ratios])
sorted_ratios[:20]


### Turn 38
def primitive_ratio(s):
    return sp.Rational(f_brutal_limited(s), s)
ratios = {}
for s in range(6, 200):
    ratios[s] = primitive_ratio(s)
sorted_ratios = sorted([(ratios[s], s) for s in ratios])
print(head(sorted_ratios, 20))

### Turn 39
def ratio_s(s):
    return sp.Rational(f_brutal_limited(s), s)

for s in [13,19,23,25,27,29,31,33,35,37,39,41]:
    print(s, ratio_s(s))


### Turn 39
def ratio_s(s):
    return sp.Rational(f_brutal_limited(s), s)
for s in [13, 19, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]:
    print(s, ratio_s(s))

### Turn 40
def f_brutal_fast(p):
    return f_brutal_limited(p)
prime_f = {}
for p in sp.primerange(5, 200):
    prime_f[p]=f_brutal_fast(p)
list(prime_f.items())[:15]


### Turn 40
def f_brutal_fast(p):
    return f_brutal_limited(p)
prime_f = {}
for p in sp.primerange(5, 200):
    prime_f[p] = f_brutal_fast(p)
print(head(list(prime_f.items()), 15))

### Turn 41
def f_brutal_small(n):
    return f_brutal_limit(n)

prime_ratios = []
for p in sp.primerange(7, 200):
    fval = f_brutal_small(p)
    prime_ratios.append((p, fval, sp.Rational(fval, p)))
prime_ratios[:20]


### Turn 41
def f_brutal_small(n):
    return f_brutal_limit(n)
prime_ratios = []
for p in sp.primerange(7, 200):
    fval = f_brutal_small(p)
    prime_ratios.append((p, fval, sp.Rational(fval, p)))
print(head(prime_ratios, 20))

### Turn 42
prime_ratios = []
for p in sp.primerange(7, 200):
    fval = f_brutal_limit(p)
    prime_ratios.append((p, fval, sp.Rational(fval, p)))
prime_ratios[:15]


### Turn 42
prime_ratios = []
for p in sp.primerange(7, 200):
    fval = f_brutal_limit(p)
    prime_ratios.append((p, fval, sp.Rational(fval, p)))
print(head(prime_ratios, 15))

### Turn 43
def f_brutal_limit_debug(n):
    return f_brutal_limited(n)
print(f_brutal_limit_debug(13))


### Turn 43
def f_brutal_limit_debug(n):
    return f_brutal_limited(n)
print(f_brutal_limit_debug(13))

### Turn 44
for p in sp.primerange(7, 30):
    print('p', p, type(p))


### Turn 44
for p in sp.primerange(7, 30):
    print('p', p, type(p))

### Turn 45
prime_ratios = []
for prime in sp.primerange(7, 200):
    fval = f_brutal_limit_debug(prime)
    prime_ratios.append((prime, fval, sp.Rational(fval, prime)))
prime_ratios[:20]


### Turn 45
prime_ratios = []
for prime in sp.primerange(7, 200):
    fval = f_brutal_limit_debug(prime)
    prime_ratios.append((prime, fval, sp.Rational(fval, prime)))
print(head(prime_ratios, 20))

### Turn 46
def floor_2p3(p):
    return (2*p)//3

for p,fval in [(prime, f_brutal_limit_debug(prime)) for prime in sp.primerange(7, 200)]:
    print(p, fval, floor_2p3(p), fval - floor_2p3(p))


### Turn 46
def floor_2p3(p):
    return 2 * p // 3
for p, fval in [(prime, f_brutal_limit_debug(prime)) for prime in sp.primerange(7, 200)]:
    print(p, fval, floor_2p3(p), fval - floor_2p3(p))

### Turn 47
import math, functools, itertools
from sympy import isprime, factorint

# Precompute f_small for n up to 35 via brute
f_small={}
for n in range(6,36):
    f_small[n]=f_brutal_limit_debug(n)
# function to compute f(n) recursively
from functools import lru_cache

@lru_cache(maxsize=None)
def f_big(n):
    # n is positive integer >=6
    if n <= 35:
        return f_small[n]
    # check if n is prime
    if isprime(n):
        # apply formula based on n mod 6
        if n % 6 == 1:
            return (2*n - 2)//3
        elif n % 6 == 5:
            return (2*n - 4)//3
        else:
            # n odd and not divisible by 3 -> must be 1 or 5 mod6
            raise ValueError("Unexpected prime modulo")
    # composite
    # generate divisors <= n/6
    # Use factorint and generate divisors
    fac = factorint(n)
    # generate all divisors via recursion
    divisors = [1]
    for p,exp in fac.items():
        new_divs=[]
        for d in divisors:
            for e in range(1,exp+1):
                new_divs.append(d*(p**e))
        divisors += new_divs
    # filter eligible divisors
    candidates=[]
    for d in divisors:
        if d>=2 and d <= n//6:
            candidates.append(d)
    # if no divisor, fallback? Should not happen for composite n>35
    best = None
    for d in candidates:
        val = d * f_big(n//d)
        if best is None or val < best:
            best = val
    # Additionally, could be that using primitive triple (no divisor) yields smaller?
    # But primitive triple ratio = f_small primitive? we need compute directly via base case.
    # For safety compute f via brute for n up to maybe 200? not needed for large n.
    # However if best is None, fallback to primitive via some known base sum? We'll attempt brute for n up to 200.
    if best is None:
        # fallback (should not occur)
        best = f_brutal_limit_debug(n)
    return best

# test function on some numbers
test_vals = [11,13,17,19,23,27,33,35,45,55,81,99]
for v in test_vals:
    print(v, f_big(v), f_small.get(v, None))


### Turn 47
import math, functools, itertools
from sympy import isprime, factorint
f_small = {}
for n in range(6, 36):
    f_small[n] = f_brutal_limit_debug(n)
from functools import lru_cache

@lru_cache(maxsize=None)
def f_big(n):
    if n <= 35:
        return f_small[n]
    if isprime(n):
        if n % 6 == 1:
            return (2 * n - 2) // 3
        elif n % 6 == 5:
            return (2 * n - 4) // 3
        else:
            raise ValueError('Unexpected prime modulo')
    fac = factorint(n)
    divisors = [1]
    for p, exp in fac.items():
        new_divs = []
        for d in divisors:
            for e in range(1, exp + 1):
                new_divs.append(d * p ** e)
        divisors += new_divs
    candidates = []
    for d in divisors:
        if d >= 2 and d <= n // 6:
            candidates.append(d)
    best = None
    for d in candidates:
        val = d * f_big(n // d)
        if best is None or val < best:
            best = val
    if best is None:
        best = f_brutal_limit_debug(n)
    return best
test_vals = [11, 13, 17, 19, 23, 27, 33, 35, 45, 55, 81, 99]
for v in test_vals:
    print(v, f_big(v), f_small.get(v, None))

### Turn 48
def min_ratio_for_c(c):
    # compute minimal ratio r = f(d)/d for d | (c+1), d>=6, 3 not divide d
    val = c+1
    divs = sp.divisors(val)
    best_ratio = None
    best_d = None
    for d in divs:
        if d < 6:
            continue
        if d % 3 == 0:
            continue
        # compute f(d)
        f_d = f_big(d)
        ratio = sp.Rational(f_d, d)
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_d = d
    return best_ratio, best_d

for c in cs_all[1:]:
    print(c, min_ratio_for_c(c))


### Turn 48
def min_ratio_for_c(c):
    val = c + 1
    divs = sp.divisors(val)
    best_ratio = None
    best_d = None
    for d in divs:
        if d < 6:
            continue
        if d % 3 == 0:
            continue
        f_d = f_big(d)
        ratio = sp.Rational(f_d, d)
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_d = d
    return (best_ratio, best_d)
for c in cs_all[1:]:
    print(c, min_ratio_for_c(c))

### Turn 49
from fractions import Fraction

sum_fraction = Fraction(0,1)
sum_fraction += Fraction(2,3)
sum_fraction += Fraction(2,3)
sum_fraction += Fraction(16,25)
sum_fraction += Fraction(30,47)
sum_fraction += Fraction(64,97)
sum_fraction += Fraction(110,167)
sum_fraction


### Turn 49
from fractions import Fraction
sum_fraction = Fraction(0, 1)
sum_fraction += Fraction(2, 3)
sum_fraction += Fraction(2, 3)
sum_fraction += Fraction(16, 25)
sum_fraction += Fraction(30, 47)
sum_fraction += Fraction(64, 97)
sum_fraction += Fraction(110, 167)
print(sum_fraction)

### Turn 50
import math
math.gcd(224414944, 57101475)


### Turn 50
import math
print(math.gcd(224414944, 57101475))

### Turn 51
(p+q) % 99991


### Turn 51
print((p + q) % 99991)

### Turn 52
p = sum_fraction.numerator
q = sum_fraction.denominator
(p+q) % 99991


### Turn 52
p = sum_fraction.numerator
q = sum_fraction.denominator
print((p + q) % 99991)

### Turn 53
from fractions import Fraction
ratios = [Fraction(2,3), Fraction(2,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)]
sum_frac = sum(ratios, Fraction(0,1))
sum_frac, sum_frac.numerator, sum_frac.denominator


### Turn 53
from fractions import Fraction
ratios = [Fraction(2, 3), Fraction(2, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)]
sum_frac = sum(ratios, Fraction(0, 1))
print((sum_frac, sum_frac.numerator, sum_frac.denominator))

### Turn 54
(p+q) % 99991


### Turn 54
print((p + q) % 99991)

### Turn 55
best30,trip30 = f_brutal_limit(45)
best30, trip30


### Turn 55
best30, trip30 = f_brutal_limit(45)
print((best30, trip30))

### Turn 56
for c in cs_all[1:]:
    print(c, c%9)


### Turn 56
for c in cs_all[1:]:
    print(c, c % 9)

