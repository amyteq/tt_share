# Auto-extracted from: P10_v9r6.md
# Attempt: 2

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
import math, itertools, sys, sympy as sp

def brute_f(n):
    # find minimal m that has 3 distinct positive divisors summing to n
    # brute up to some limit? We'll search for triples of divisors a<b<c positive sum n.
    # Since we need m to have a,b,c as divisors, smallest m is lcm(a,b,c).
    best = None
    for a in range(1, n-2):
        for b in range(a+1, n-1):
            c = n - a - b
            if c <= b:
                continue
            # distinct a,b,c >0
            # compute lcm
            m = sp.ilcm(a, b, c)
            if best is None or m < best:
                best = m
    return best

def formula_f(n):
    s = n-1
    if sp.isprime(s):
        return 2*s - 4
    # find smallest divisor >2
    # find all divisors of s
    # We'll get smallest divisor >2
    # Use sympy divisor list maybe
    for d in range(3, int(math.isqrt(s)) + 1):
        if s % d == 0:
            k = d
            break
    else:
        # no divisor between 3 and sqrt(s), so s is power of 2? Actually could be s=2^a*?? But if not prime, must have divisor >2; if it's power of 2, smallest divisor >2 is 4.
        if s % 4 == 0:
            k = 4
        else:
            # s might be 2 * p where p prime > 2; then smallest divisor >2 is p
            # find p by dividing out factor 2
            if s % 2 == 0:
                # find p = s//2
                p = s // 2
                if sp.isprime(p):
                    k = p
                else:
                    # should not happen
                    raise ValueError('unexpected')
            else:
                # not prime but no divisor up to sqrt?? improbable.
                k = s
    return s * (k-1) // k

# test
for n in range(6,101):
    bf = brute_f(n)
    ff = formula_f(n)
    if bf != ff:
        print(n, bf, ff)
        break
else:
    print("All good up to 100")


### Turn 0
import math, itertools, sys, sympy as sp

def brute_f(n):
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - 1):
            c = n - a - b
            if c <= b:
                continue
            m = sp.ilcm(a, b, c)
            if best is None or m < best:
                best = m
    return best

def formula_f(n):
    s = n - 1
    if sp.isprime(s):
        return 2 * s - 4
    for d in range(3, int(math.isqrt(s)) + 1):
        if s % d == 0:
            k = d
            break
    else:
        if s % 4 == 0:
            k = 4
        elif s % 2 == 0:
            p = s // 2
            if sp.isprime(p):
                k = p
            else:
                raise ValueError('unexpected')
        else:
            k = s
    return s * (k - 1) // k
for n in range(6, 101):
    bf = brute_f(n)
    ff = formula_f(n)
    if bf != ff:
        print(n, bf, ff)
        break
else:
    print('All good up to 100')

### Turn 1
import math
mod=11
pow(3,5,mod)


### Turn 1
import math
mod = 11
print(pow(3, 5, mod))

### Turn 2
cs = [0, None, 1848374, 10162574, 265710644, 44636594]
# compute c for case 4M later separately
for c in cs:
    if c is not None:
        print(c, c%11)


### Turn 2
cs = [0, None, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    if c is not None:
        print(c, c % 11)

### Turn 3
1848374 % 5


### Turn 3
print(1848374 % 5)

### Turn 4
cs = [0, 4*pow(3, math.factorial(2025), 10**9), 1848374, 10162574, 265710644, 44636594] # Wait we cannot compute 4M? We'll treat mod 14.
def mod_c(c):
    return c % 14

cs_simple = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 14 for c in cs_simple]
mods


### Turn 4
cs = [0, 4 * pow(3, math.factorial(2025), 10 ** 9), 1848374, 10162574, 265710644, 44636594]

def mod_c(c):
    return c % 14
cs_simple = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 14 for c in cs_simple]
print(mods)

### Turn 5
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c%19)


### Turn 5
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 19)

### Turn 6
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c%20)


### Turn 6
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 20)

### Turn 7
def order_mod(a, mod):
    for k in range(1, mod):
        if pow(a,k,mod)==1:
            return k
    return None

order_mod(3,23)


### Turn 7
def order_mod(a, mod):
    for k in range(1, mod):
        if pow(a, k, mod) == 1:
            return k
    return None
print(order_mod(3, 23))

### Turn 8
for c in cs:
    print(c, c%23)


### Turn 8
for c in cs:
    print(c, c % 23)

### Turn 9
import math, sympy as sp

def candidate_f(n):
    s = n - 1
    # L1 via divisor of s
    L1 = None
    # find all divisors of s
    divs = sp.divisors(s)
    # find d in divs where 2 <= d <= s-2 and d != s/2
    for d in divs:
        if d >= 2 and d <= s-2 and not (s % 2 == 0 and d == s//2):
            # compute L = s - d
            L = s - d
            if L1 is None or L < L1:
                L1 = L
    # L2 via t dividing n, t = pq - 1
    L2 = None
    # get divisors of n
    divns = sp.divisors(n)
    for t in divns:
        if t <= 0:
            continue
        # t must be >=11 at least? We'll just try p,q >2 distinct.
        # Find possible p,q such that pq - 1 = t.
        # Iterate p from 3 to sqrt(t+1)
        for p in range(3, int(math.isqrt(t+1)) + 2):
            q_candidate = (t+1) // p
            if p * q_candidate == t + 1 and q_candidate > 2 and p != q_candidate:
                p_, q_ = p, q_candidate
                # compute ratio
                L = n * ((p_-1)*(q_-1)) // t  # integer because divisor
                # ensure integer division exact
                if L2 is None or L < L2:
                    L2 = L
    # return min of available L1, L2
    candidates = [v for v in [L1, L2] if v is not None]
    if candidates:
        return min(candidates)
    else:
        # fallback to brute
        return None

def brute_f(n):
    best = None
    for a in range(1, n-2):
        for b in range(a+1, n-1):
            c = n - a - b
            if c <= b:
                continue
            # distinct a,b,c automatically
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
    return best

diff = []
for n in range(6,201):
    bf = brute_f(n)
    cf = candidate_f(n)
    if bf != cf:
        diff.append((n,bf,cf))
len(diff), diff[:10]


### Turn 9
import math, sympy as sp

def candidate_f(n):
    s = n - 1
    L1 = None
    divs = sp.divisors(s)
    for d in divs:
        if d >= 2 and d <= s - 2 and (not (s % 2 == 0 and d == s // 2)):
            L = s - d
            if L1 is None or L < L1:
                L1 = L
    L2 = None
    divns = sp.divisors(n)
    for t in divns:
        if t <= 0:
            continue
        for p in range(3, int(math.isqrt(t + 1)) + 2):
            q_candidate = (t + 1) // p
            if p * q_candidate == t + 1 and q_candidate > 2 and (p != q_candidate):
                p_, q_ = (p, q_candidate)
                L = n * ((p_ - 1) * (q_ - 1)) // t
                if L2 is None or L < L2:
                    L2 = L
    candidates = [v for v in [L1, L2] if v is not None]
    if candidates:
        return min(candidates)
    else:
        return None

def brute_f(n):
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - 1):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
    return best
diff = []
for n in range(6, 201):
    bf = brute_f(n)
    cf = candidate_f(n)
    if bf != cf:
        diff.append((n, bf, cf))
print((len(diff), head(diff, 10)))

### Turn 10
def min_triple(n):
    best = (None, None, None)
    bestL = None
    for a in range(1,n-2):
        for b in range(a+1, n-1):
            c=n-a-b
            if c<=b:
                continue
            L=sp.ilcm(a,b,c)
            if bestL is None or L<bestL:
                bestL=L
                best=(a,b,c)
    return bestL,best

print(min_triple(21))


### Turn 10
def min_triple(n):
    best = (None, None, None)
    bestL = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - 1):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if bestL is None or L < bestL:
                bestL = L
                best = (a, b, c)
    return (bestL, best)
print(min_triple(21))

### Turn 11
def min_ratio(limit):
    best_ratio = 1
    best_triple = None
    for a in range(1, limit):
        for b in range(a+1, limit):
            for c in range(b+1, limit*2):  # maybe large
                S = a+b+c
                if S>limit: 
                    continue
                L = sp.ilcm(a,b,c)
                ratio = L / S
                if ratio < best_ratio - 1e-12:
                    best_ratio = ratio
                    best_triple = (a,b,c)
    return best_ratio, best_triple

best_ratio, best_triple = min_ratio(30)
best_ratio, best_triple


### Turn 11
def min_ratio(limit):
    best_ratio = 1
    best_triple = None
    for a in range(1, limit):
        for b in range(a + 1, limit):
            for c in range(b + 1, limit * 2):
                S = a + b + c
                if S > limit:
                    continue
                L = sp.ilcm(a, b, c)
                ratio = L / S
                if ratio < best_ratio - 1e-12:
                    best_ratio = ratio
                    best_triple = (a, b, c)
    return (best_ratio, best_triple)
best_ratio, best_triple = min_ratio(30)
print((best_ratio, best_triple))

### Turn 12
def min_ratio_upto(N):
    best_ratio = 1
    best = None
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                S = a+b+c
                L = sp.ilcm(a,b,c)
                ratio = L / S
                if ratio < best_ratio:
                    best_ratio = ratio
                    best = (a,b,c)
    return best_ratio, best

best_ratio, best_trip = min_ratio_upto(200)
best_ratio, best_trip


### Turn 12
def min_ratio_upto(N):
    best_ratio = 1
    best = None
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                S = a + b + c
                L = sp.ilcm(a, b, c)
                ratio = L / S
                if ratio < best_ratio:
                    best_ratio = ratio
                    best = (a, b, c)
    return (best_ratio, best)
best_ratio, best_trip = min_ratio_upto(200)
print((best_ratio, best_trip))

### Turn 13
def min_ratio_xy(N):
    best = 1
    best_pair = None
    for x in range(2,N+1):
        for y in range(x+1,N+1):
            if math.gcd(x,y)!=1:
                continue
            r = x*y/(x+y+x*y)
            if r < best:
                best = r
                best_pair = (x,y)
    return best, best_pair

best_r, pair = min_ratio_xy(50)
best_r, pair


### Turn 13
def min_ratio_xy(N):
    best = 1
    best_pair = None
    for x in range(2, N + 1):
        for y in range(x + 1, N + 1):
            if math.gcd(x, y) != 1:
                continue
            r = x * y / (x + y + x * y)
            if r < best:
                best = r
                best_pair = (x, y)
    return (best, best_pair)
best_r, pair = min_ratio_xy(50)
print((best_r, pair))

### Turn 14
def generate_t(limit):
    ts=set()
    for x in range(2,limit+1):
        for y in range(x+1,limit+1):
            if math.gcd(x,y)!=1:
                continue
            t = (x+1)*(y+1)-1
            ts.add(t)
    return sorted(ts)

ts = generate_t(30)
ts[:20], len(ts)


### Turn 14
def generate_t(limit):
    ts = set()
    for x in range(2, limit + 1):
        for y in range(x + 1, limit + 1):
            if math.gcd(x, y) != 1:
                continue
            t = (x + 1) * (y + 1) - 1
            ts.add(t)
    return sorted(ts)
ts = generate_t(30)
print((head(ts, 20), len(ts)))

### Turn 15
1848374 % 11


### Turn 15
print(1848374 % 11)

### Turn 16
10162574 % 11


### Turn 16
print(10162574 % 11)

### Turn 17
265710644 % 11


### Turn 17
print(265710644 % 11)

### Turn 18
44636594 % 11


### Turn 18
print(44636594 % 11)

### Turn 19
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    r = (c+1) % 11
    print(c, r)


### Turn 19
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    r = (c + 1) % 11
    print(c, r)

### Turn 20
def f_brute(n):
    return brute_f(n)
def f_estimate(n):
    # floor(6n/11) maybe?
    return (6*n)//11

diffs=[]
for n in range(6,200):
    bf=f_brute(n)
    est=(6*n)//11
    if bf != est:
        diffs.append((n,bf,est))
len(diffs), diffs[:20]


### Turn 20
def f_brute(n):
    return brute_f(n)

def f_estimate(n):
    return 6 * n // 11
diffs = []
for n in range(6, 200):
    bf = f_brute(n)
    est = 6 * n // 11
    if bf != est:
        diffs.append((n, bf, est))
print((len(diffs), head(diffs, 20)))

### Turn 21
ratios = [(n, f_brute(n)/n) for n in range(6,101)]
min(ratios, key=lambda x:x[1])


### Turn 21
ratios = [(n, f_brute(n) / n) for n in range(6, 101)]
print(min(ratios, key=lambda x: x[1]))

### Turn 22
print(f_brute(22), f_brute(22)/22)


### Turn 22
print(f_brute(22), f_brute(22) / 22)

### Turn 23
print(f_brute(33), f_brute(33)/33)


### Turn 23
print(f_brute(33), f_brute(33) / 33)

### Turn 24
from collections import defaultdict
rem_best = defaultdict(list)
for n in range(6,200):
    bf = f_brute(n)
    rem = n % 11
    rem_best[rem].append((n,bf))
for r in range(11):
    # find smallest n with each remainder and ratio
    mins = min(rem_best[r], key=lambda x:x[1]/x[0]) if rem_best[r] else None
    if mins:
        print(r, mins, mins[1]/mins[0])


### Turn 24
from collections import defaultdict
rem_best = defaultdict(list)
for n in range(6, 200):
    bf = f_brute(n)
    rem = n % 11
    rem_best[rem].append((n, bf))
for r in range(11):
    mins = min(rem_best[r], key=lambda x: x[1] / x[0]) if rem_best[r] else None
    if mins:
        print(r, mins, mins[1] / mins[0])

### Turn 25
def ratio(n):
    return f_brute(n)/n

bad = []
for n in range(6,200):
    r = ratio(n)
    if n%11!=0 and r > 4/7 + 1e-12:
        bad.append((n, r))
bad[:20], len(bad)


### Turn 25
def ratio(n):
    return f_brute(n) / n
bad = []
for n in range(6, 200):
    r = ratio(n)
    if n % 11 != 0 and r > 4 / 7 + 1e-12:
        bad.append((n, r))
print((head(bad, 20), len(bad)))

### Turn 26
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 7)


### Turn 26
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 7)

### Turn 27
1848374 % 3


### Turn 27
print(1848374 % 3)

### Turn 28
10162574 % 3


### Turn 28
print(10162574 % 3)

### Turn 29
265710644 % 3


### Turn 29
print(265710644 % 3)

### Turn 30
44636594 % 3


### Turn 30
print(44636594 % 3)

### Turn 31
print('f(27)=', brute_f(27))


### Turn 31
print('f(27)=', brute_f(27))

### Turn 32
def best_triple(n):
    bestL = None
    best = None
    for a in range(1,n-2):
        for b in range(a+1, n-1):
            c = n-a-b
            if c <= b:
                continue
            L=sp.ilcm(a,b,c)
            if bestL is None or L<bestL:
                bestL=L
                best=(a,b,c)
    return bestL,best

print(best_triple(27))


### Turn 32
def best_triple(n):
    bestL = None
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - 1):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if bestL is None or L < bestL:
                bestL = L
                best = (a, b, c)
    return (bestL, best)
print(best_triple(27))

### Turn 33
for c in cs:
    print(c, c%9)


### Turn 33
for c in cs:
    print(c, c % 9)

### Turn 34
def min_ratio_for_sum(S):
    best = None
    best_trip = None
    for a in range(1,S-1):
        for b in range(a+1,S):
            c = S - a - b
            if c <= b:
                continue
            L = sp.ilcm(a,b,c)
            r = L / S
            if best is None or r < best:
                best = r
                best_trip = (a,b,c)
    return best, best_trip

powers = [3,9,27,81,243]
for S in powers:
    best,trip = min_ratio_for_sum(S)
    print(S, best, trip)


### Turn 34
def min_ratio_for_sum(S):
    best = None
    best_trip = None
    for a in range(1, S - 1):
        for b in range(a + 1, S):
            c = S - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            r = L / S
            if best is None or r < best:
                best = r
                best_trip = (a, b, c)
    return (best, best_trip)
powers = [3, 9, 27, 81, 243]
for S in powers:
    best, trip = min_ratio_for_sum(S)
    print(S, best, trip)

### Turn 35
import sympy as sp
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c+1))


### Turn 35
import sympy as sp
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c + 1))

### Turn 36
def best_ratio_sum(S):
    best = 2 # >1
    best_trip = None
    for a in range(1, S-1):
        for b in range(a+1, S):
            c = S - a - b
            if c <= b:
                continue
            L = sp.ilcm(a,b,c)
            r = L / S
            if r < best:
                best = r
                best_trip = (a,b,c)
    return best, best_trip

for S in range(6,31):
    if S % 3 !=0: # skip multiples of 3 maybe
        br,_ = best_ratio_sum(S)
        # print S if ratio less than previous best
        print(S, br)


### Turn 36
def best_ratio_sum(S):
    best = 2
    best_trip = None
    for a in range(1, S - 1):
        for b in range(a + 1, S):
            c = S - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            r = L / S
            if r < best:
                best = r
                best_trip = (a, b, c)
    return (best, best_trip)
for S in range(6, 31):
    if S % 3 != 0:
        br, _ = best_ratio_sum(S)
        print(S, br)

### Turn 37
for c in cs:
    print(c, (c+1)%11)


### Turn 37
for c in cs:
    print(c, (c + 1) % 11)

### Turn 38
for c in cs:
    print(c, (c+1)%7)


### Turn 38
for c in cs:
    print(c, (c + 1) % 7)

### Turn 39
for c in cs:
    print(c, (c+1)%13)


### Turn 39
for c in cs:
    print(c, (c + 1) % 13)

### Turn 40
def min_ratio_c(c):
    # compute divisors of c+1 excluding multiples of 3
    N = c+1
    divs = sp.divisors(N)
    # filter
    candidates = [d for d in divs if d>=6 and d%3!=0]
    best_ratio = 2 # >1
    best_S = None
    best_trip = None
    for S in candidates:
        # compute minimal ratio for sum S via brute up to S (if S not huge)
        # If S too big, we can maybe use known formulas: if S divisible by 11 then ratio = 6/11.
        # else if S divisible by 7 -> 4/7.
        # else compute via brute maybe but S will be moderate
        if S % 11 == 0:
            ratio = 6/11
            # we can break early since that's minimal possible overall (unless 6/11 cannot be beaten)
            # but we need check other sums maybe also 6/11; same.
            return ratio, S, '11-divisible'
        if S % 7 == 0:
            ratio = 4/7
            # maybe better than current best
            if ratio < best_ratio:
                best_ratio = ratio
                best_S = S
                best_trip = '7-divisible'
            continue
        # else compute brute minimal ratio for sum S up to moderate size
        # Use brute only if S <= 200 maybe to avoid heavy.
        if S <= 200:
            br,trip = best_ratio_sum(S)
            if br < best_ratio:
                best_ratio = br
                best_S = S
                best_trip = trip
        else:
            # for larger S not divisible by 11 or 7, we don't compute brute due to cost.
            # approximate ratio maybe > 0.6? Not minimal.
            pass
    return best_ratio, best_S, best_trip

for c in cs:
    print(c, min_ratio_c(c))


### Turn 40
def min_ratio_c(c):
    N = c + 1
    divs = sp.divisors(N)
    candidates = [d for d in divs if d >= 6 and d % 3 != 0]
    best_ratio = 2
    best_S = None
    best_trip = None
    for S in candidates:
        if S % 11 == 0:
            ratio = 6 / 11
            return (ratio, S, '11-divisible')
        if S % 7 == 0:
            ratio = 4 / 7
            if ratio < best_ratio:
                best_ratio = ratio
                best_S = S
                best_trip = '7-divisible'
            continue
        if S <= 200:
            br, trip = best_ratio_sum(S)
            if br < best_ratio:
                best_ratio = br
                best_S = S
                best_trip = trip
        else:
            pass
    return (best_ratio, best_S, best_trip)
for c in cs:
    print(c, min_ratio_c(c))

### Turn 41
br,trip = best_ratio_sum(25)
br,trip


### Turn 41
br, trip = best_ratio_sum(25)
print((br, trip))

### Turn 42
c=1848374
N=c+1
max_div = max([d for d in sp.divisors(N) if d%3!=0])
max_div


### Turn 42
c = 1848374
N = c + 1
max_div = max([d for d in sp.divisors(N) if d % 3 != 0])
print(max_div)

### Turn 43
def min_ratio_k_l(limit):
    best = 2
    best_pair = None
    for k in range(2, limit+1):
        for m in range(2, limit+1):
            l = k*m
            if l <= k:
                continue
            r = l / (1 + k + l)
            if r < best:
                best = r
                best_pair = (k,l)
    return best, best_pair

min_ratio_k_l(20)


### Turn 43
def min_ratio_k_l(limit):
    best = 2
    best_pair = None
    for k in range(2, limit + 1):
        for m in range(2, limit + 1):
            l = k * m
            if l <= k:
                continue
            r = l / (1 + k + l)
            if r < best:
                best = r
                best_pair = (k, l)
    return (best, best_pair)
print(min_ratio_k_l(20))

### Turn 44
1848374+1


### Turn 44
print(1848374 + 1)

### Turn 45
br,trip = best_ratio_sum(31)
br, trip


### Turn 45
br, trip = best_ratio_sum(31)
print((br, trip))

### Turn 46
def min_ratio_for_c(c, limit=5000):
    N = c+1
    # generate all divisors up to limit
    divs = sp.divisors(N)
    # filter for <= limit and not divisible by 3
    candidate_S = [d for d in divs if d <= limit and d >= 6 and d % 3 != 0]
    # also include maybe S that are multiples of small prime but > limit? We'll ignore for now
    best_ratio = None
    best_S = None
    best_info = None
    # precompute best ratio for sums up to limit via brute
    # We'll compute a dictionary
    brute_cache = {}
    for S in candidate_S:
        # if divisible by 11
        if S % 11 == 0:
            ratio = 6/11
            info = '11'
        elif S % 7 == 0:
            ratio = 4/7
            info = '7'
        else:
            # compute via brute if not in cache
            if S not in brute_cache:
                br,_ = best_ratio_sum(S)
                brute_cache[S] = br
            ratio = brute_cache[S]
            info = 'brute'
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_S = S
            best_info = info
    # If best_ratio still None (no small divisor), maybe we need to consider larger divisors using (1,a,2a) pattern.
    if best_ratio is None:
        # fallback: use (1,a,2a) pattern with a from N? not possible
        # For large S not multiple of 3, the ratio maybe > 0.8
        # We'll approximate ratio as 0.8 (from 4/5) from type A? Not needed likely.
        pass
    return best_ratio, best_S, best_info

for c in cs[1:]:
    ratio, S, info = min_ratio_for_c(c, limit=5000)
    print(c, ratio, S, info)


### Turn 46
def min_ratio_for_c(c, limit=5000):
    N = c + 1
    divs = sp.divisors(N)
    candidate_S = [d for d in divs if d <= limit and d >= 6 and (d % 3 != 0)]
    best_ratio = None
    best_S = None
    best_info = None
    brute_cache = {}
    for S in candidate_S:
        if S % 11 == 0:
            ratio = 6 / 11
            info = '11'
        elif S % 7 == 0:
            ratio = 4 / 7
            info = '7'
        else:
            if S not in brute_cache:
                br, _ = best_ratio_sum(S)
                brute_cache[S] = br
            ratio = brute_cache[S]
            info = 'brute'
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_S = S
            best_info = info
    if best_ratio is None:
        pass
    return (best_ratio, best_S, best_info)
for c in cs[1:]:
    ratio, S, info = min_ratio_for_c(c, limit=5000)
    print(c, ratio, S, info)

### Turn 47
def v_p_factorial(n,p):
    count=0
    power=p
    while power<=n:
        count+=n//power
        power*=p
    return count

v5 = v_p_factorial(2025,5)
v5


### Turn 47
def v_p_factorial(n, p):
    count = 0
    power = p
    while power <= n:
        count += n // power
        power *= p
    return count
v5 = v_p_factorial(2025, 5)
print(v5)

### Turn 48
from fractions import Fraction
terms = [Fraction(2,3), Fraction(10,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)]
S = sum(terms, Fraction(0,1))
S, S.numerator, S.denominator


### Turn 48
from fractions import Fraction
terms = [Fraction(2, 3), Fraction(10, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)]
S = sum(terms, Fraction(0, 1))
print((S, S.numerator, S.denominator))

### Turn 49
import math
math.gcd(125561848, 19033825)


### Turn 49
import math
print(math.gcd(125561848, 19033825))

### Turn 50
p=125561848
q=19033825
mod=99991
(p+q) % mod


### Turn 50
p = 125561848
q = 19033825
mod = 99991
print((p + q) % mod)

### Turn 51
def min_ratio_by_candidates(S):
    # compute minimal ratio for sum S using candidate methods
    best = None
    # candidate 1: if S%11==0 ratio=6/11
    if S % 11 == 0:
        best = 6/11
    # candidate 2: if S%7==0 ratio=4/7
    if S % 7 == 0:
        r = 4/7
        if best is None or r < best:
            best = r
    # candidate 3: if S %3 == 1 (S=3a+1) and a>1
    if S % 3 == 1:
        a = (S - 1)//3
        if a > 1:
            r = (2*a)/S
            if best is None or r < best:
                best = r
    # candidate 4: factor S+1 into p*q with p,q>2 distinct
    for p in range(3, int(math.isqrt(S+1)) + 1):
        if (S+1) % p == 0:
            q = (S+1)//p
            if q > 2 and p != q:
                r = ((p-1)*(q-1)) / S
                if best is None or r < best:
                    best = r
    # candidate 5: type A using smallest divisor >2 of S
    # Find smallest divisor >2 of S (odd divisor)
    # We can factor S quickly using sympy factorint
    fac = sp.factorint(S)
    # Find smallest divisor >2
    min_div = None
    for d in sp.divisors(S):
        if d > 2 and d < S:
            if min_div is None or d < min_div:
                # ensure not equal to S/2 maybe?
                if not (S % 2 == 0 and d == S//2):
                    min_div = d
    if min_div is not None:
        r = 1 - 1/min_div
        if best is None or r < best:
            best = r
    return best

def min_ratio_over_divisors(N):
    # N = c+1
    best = None
    best_S=None
    for d in sp.divisors(N):
        if d >= 6 and d%3 != 0: # exclude multiples of 3? actually we can include but handle
            r = min_ratio_by_candidates(d)
            if r is not None:
                if best is None or r < best:
                    best = r
                    best_S = d
    return best,best_S

for c in cs[1:]:
    N = c+1
    best,bS = min_ratio_over_divisors(N)
    print(c, best, bS)


### Turn 51
def min_ratio_by_candidates(S):
    best = None
    if S % 11 == 0:
        best = 6 / 11
    if S % 7 == 0:
        r = 4 / 7
        if best is None or r < best:
            best = r
    if S % 3 == 1:
        a = (S - 1) // 3
        if a > 1:
            r = 2 * a / S
            if best is None or r < best:
                best = r
    for p in range(3, int(math.isqrt(S + 1)) + 1):
        if (S + 1) % p == 0:
            q = (S + 1) // p
            if q > 2 and p != q:
                r = (p - 1) * (q - 1) / S
                if best is None or r < best:
                    best = r
    fac = sp.factorint(S)
    min_div = None
    for d in sp.divisors(S):
        if d > 2 and d < S:
            if min_div is None or d < min_div:
                if not (S % 2 == 0 and d == S // 2):
                    min_div = d
    if min_div is not None:
        r = 1 - 1 / min_div
        if best is None or r < best:
            best = r
    return best

def min_ratio_over_divisors(N):
    best = None
    best_S = None
    for d in sp.divisors(N):
        if d >= 6 and d % 3 != 0:
            r = min_ratio_by_candidates(d)
            if r is not None:
                if best is None or r < best:
                    best = r
                    best_S = d
    return (best, best_S)
for c in cs[1:]:
    N = c + 1
    best, bS = min_ratio_over_divisors(N)
    print(c, best, bS)

### Turn 52
from fractions import Fraction
sum_others = sum([Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)], Fraction(0,1))
total = Fraction(4,1) + sum_others
total, total.numerator, total.denominator


### Turn 52
from fractions import Fraction
sum_others = sum([Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)], Fraction(0, 1))
total = Fraction(4, 1) + sum_others
print((total, total.numerator, total.denominator))

### Turn 53
(p+q) % 99991


### Turn 53
print((p + q) % 99991)

