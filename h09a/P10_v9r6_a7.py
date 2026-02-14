# Auto-extracted from: P10_v9r6.md
# Attempt: 7

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
    min_lcm = None
    # consider distinct positive ints a<b<c maybe not order needed.
    # To limit, we can just iterate a from 1..n-2, b from a+1..n-1-a, c = n-a-b, ensure distinct.
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            # distinct automatically because a<b<c
            # compute lcm
            l = math.lcm(a,b,c)
            if min_lcm is None or l < min_lcm:
                min_lcm = l
    return min_lcm

def f_brut_all(nmax):
    vals = {}
    for n in range(6, nmax+1):
        vals[n] = f_brut(n)
    return vals

vals = f_brut_all(100)
vals[:10] # just test

### Turn 0
import math, itertools

def f_brut(n):
    min_lcm = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if min_lcm is None or l < min_lcm:
                min_lcm = l
    return min_lcm

def f_brut_all(nmax):
    vals = {}
    for n in range(6, nmax + 1):
        vals[n] = f_brut(n)
    return vals
vals = f_brut_all(100)
print(head(vals, 10))

### Turn 1
for n in range(6, 31):
    print(n, vals[n])


### Turn 1
for n in range(6, 31):
    print(n, vals[n])

### Turn 2
def f_candidate(n):
    # candidate via 1,d,c
    n1 = n-1
    # find largest divisor d of n1, d>1 and d != n1/2
    best = None
    for d in range(2, n1//2+1):
        if n1 % d == 0:
            if 2*d == n1:
                continue
            best = d
    if best is None:
        # fallback maybe using other approach: use odd divisor method
        # compute smallest odd divisor >=7 of n
        # but for now return None
        return None
    return n1 - best

for n in range(6,31):
    print(n, vals[n], f_candidate(n))


### Turn 2
def f_candidate(n):
    n1 = n - 1
    best = None
    for d in range(2, n1 // 2 + 1):
        if n1 % d == 0:
            if 2 * d == n1:
                continue
            best = d
    if best is None:
        return None
    return n1 - best
for n in range(6, 31):
    print(n, vals[n], f_candidate(n))

### Turn 3
def f_candidate2(n):
    # find smallest odd divisor d of n such that d >= 7
    # (including maybe d=n if n itself odd >=7)
    # then return n*(d-3)//d (must be integer)
    for d in range(7, n+1, 2):
        if n % d == 0:
            return n * (d-3) // d
    return None

for n in range(6,31):
    print(n, vals[n], f_candidate2(n))


### Turn 3
def f_candidate2(n):
    for d in range(7, n + 1, 2):
        if n % d == 0:
            return n * (d - 3) // d
    return None
for n in range(6, 31):
    print(n, vals[n], f_candidate2(n))

### Turn 4
def f_estimate(n):
    vals = []
    # candidate1 using divisor of n-1 (if exists)
    n1 = n-1
    best = 0
    best_d = None
    for d in range(2, n1//2+1):
        if n1 % d == 0 and 2*d != n1:
            if d > best:
                best = d
                best_d = d
    if best_d:
        vals.append(n1 - best_d)
    # candidate2 using odd divisor >=7
    for d in range(7, n+1, 2):
        if n % d == 0:
            vals.append(n * (d-3) // d)
            break
    if vals:
        return min(vals)
    else:
        # no candidate from these methods, maybe fallback to something else?
        return None

for n in range(6,51):
    print(n, vals[n], f_estimate(n))


### Turn 4
def f_estimate(n):
    vals = []
    n1 = n - 1
    best = 0
    best_d = None
    for d in range(2, n1 // 2 + 1):
        if n1 % d == 0 and 2 * d != n1:
            if d > best:
                best = d
                best_d = d
    if best_d:
        vals.append(n1 - best_d)
    for d in range(7, n + 1, 2):
        if n % d == 0:
            vals.append(n * (d - 3) // d)
            break
    if vals:
        return min(vals)
    else:
        return None
for n in range(6, 51):
    print(n, vals[n], f_estimate(n))

### Turn 5
c = 1848374
c%3


### Turn 5
c = 1848374
print(c % 3)

### Turn 6
10162574%3


### Turn 6
print(10162574 % 3)

### Turn 7
265710644%3


### Turn 7
print(265710644 % 3)

### Turn 8
44636594%3


### Turn 8
print(44636594 % 3)

### Turn 9
import math, itertools

def f_brut_small(n):
    # compute minimal lcm via enumeration of a,b,c distinct positive ints summing to n
    min_l = None
    for a in range(1, n//3+1):
        for b in range(a+1, (n-a)//2+1):
            c = n - a - b
            if c <= b:
                continue
            l = math.lcm(a,b,c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_estimate_general(n):
    # compute candidate using divisor of n-1 (odd prime factor minimal)
    # we compute smallest odd prime divisor p of n-1 (p != 2)
    n1 = n-1
    # find smallest odd prime divisor
    p = None
    # we exclude p=2 because we cannot use d=n1/2 (duplicate)
    # find smallest odd prime divisor of n1
    # Actually if n1 even, we need smallest odd prime divisor >2 dividing n1; else if n1 odd, then smallest prime divisor itself maybe odd.
    # We'll just search primes.
    for i in range(3, int(math.sqrt(n1))+1, 2):
        if n1 % i == 0:
            p = i
            break
    if p is None:
        # n1 is prime or power of 2? If n1 prime odd then p=n1
        if n1 % 2 == 1:
            p = n1
        else:
            # n1 is power of 2: then smallest odd prime factor doesn't exist
            p = None
    cand1 = None
    if p is not None:
        cand1 = (n-1)*(p-1)//p
    # candidate2 using smallest odd divisor d of n >=7
    cand2 = None
    # find smallest odd divisor >=7 of n
    for d in range(7, int(math.sqrt(n))+1, 2):
        if n % d == 0:
            # then d divides n
            cand2 = n * (d-3) // d
            break
    # If no divisor found, maybe n itself (odd) is >=7, then d=n works: gives n*(n-3)/n = n-3.
    if cand2 is None and n % 2 == 1 and n >= 7:
        cand2 = n-3  # using d=n
    # also possible d=9 if n divisible by 9 etc.
    # compute min non-none
    candidates = [c for c in [cand1,cand2] if c is not None]
    if not candidates:
        # fallback use n perhaps
        return n
    return min(candidates)

def test_patterns(k, c_list):
    M = 3**k
    for c in c_list:
        n = M + c
        f_true = f_brut_small(n) if n<5000 else None
        est = f_estimate_general(n)
        print("k",k,"c",c,"n",n,"est",est, "true", f_true)

test_patterns(6, [0,1,2,4,5,6,7,8,9])


### Turn 9
import math, itertools

def f_brut_small(n):
    min_l = None
    for a in range(1, n // 3 + 1):
        for b in range(a + 1, (n - a) // 2 + 1):
            c = n - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_estimate_general(n):
    n1 = n - 1
    p = None
    for i in range(3, int(math.sqrt(n1)) + 1, 2):
        if n1 % i == 0:
            p = i
            break
    if p is None:
        if n1 % 2 == 1:
            p = n1
        else:
            p = None
    cand1 = None
    if p is not None:
        cand1 = (n - 1) * (p - 1) // p
    cand2 = None
    for d in range(7, int(math.sqrt(n)) + 1, 2):
        if n % d == 0:
            cand2 = n * (d - 3) // d
            break
    if cand2 is None and n % 2 == 1 and (n >= 7):
        cand2 = n - 3
    candidates = [c for c in [cand1, cand2] if c is not None]
    if not candidates:
        return n
    return min(candidates)

def test_patterns(k, c_list):
    M = 3 ** k
    for c in c_list:
        n = M + c
        f_true = f_brut_small(n) if n < 5000 else None
        est = f_estimate_general(n)
        print('k', k, 'c', c, 'n', n, 'est', est, 'true', f_true)
print(test_patterns(6, [0, 1, 2, 4, 5, 6, 7, 8, 9]))

### Turn 10
def compute_f_range(start, end):
    res = {}
    for n in range(start, end+1):
        res[n] = f_brut_small(n)
    return res

# Let's compute around M=3^6=729 from 720 to 750
vals_range = compute_f_range(720, 750)
for n in range(720, 751):
    print(n, vals_range[n])


### Turn 10
def compute_f_range(start, end):
    res = {}
    for n in range(start, end + 1):
        res[n] = f_brut_small(n)
    return res
vals_range = compute_f_range(720, 750)
for n in range(720, 751):
    print(n, vals_range[n])

### Turn 11
def find_triple(n, target_lcm):
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            if math.lcm(a,b,c) == target_lcm:
                return (a,b,c)
    return None

print(find_triple(720, 432))


### Turn 11
def find_triple(n, target_lcm):
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            if math.lcm(a, b, c) == target_lcm:
                return (a, b, c)
    return None
print(find_triple(720, 432))

### Turn 12
print(find_triple(726, 396))


### Turn 12
print(find_triple(726, 396))

### Turn 13
print(find_triple(734, 488))


### Turn 13
print(find_triple(734, 488))

### Turn 14
def r_min_S(S):
    min_r = None
    best = None
    for a in range(1, S):
        for b in range(a+1, S):
            c = S - a - b
            if c <= b:
                continue
            l = math.lcm(a,b,c)
            r = l / S
            if min_r is None or r < min_r:
                min_r = r
                best = (a,b,c,l)
    return min_r, best

rvals = {}
for S in range(6, 31):
    r, triple = r_min_S(S)
    rvals[S] = (r, triple)
    print(S, r, triple)


### Turn 14
def r_min_S(S):
    min_r = None
    best = None
    for a in range(1, S):
        for b in range(a + 1, S):
            c = S - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            r = l / S
            if min_r is None or r < min_r:
                min_r = r
                best = (a, b, c, l)
    return (min_r, best)
rvals = {}
for S in range(6, 31):
    r, triple = r_min_S(S)
    rvals[S] = (r, triple)
    print(S, r, triple)

### Turn 15
def compute_global_min(limit):
    global_min = 1e9
    min_S = None
    min_triple = None
    for S in range(6, limit+1):
        r, triple = r_min_S(S)
        if r < global_min:
            global_min = r
            min_S = S
            min_triple = triple
    return global_min, min_S, min_triple

global_min, min_S, triple = compute_global_min(200)
global_min, min_S, triple


### Turn 15
def compute_global_min(limit):
    global_min = 1000000000.0
    min_S = None
    min_triple = None
    for S in range(6, limit + 1):
        r, triple = r_min_S(S)
        if r < global_min:
            global_min = r
            min_S = S
            min_triple = triple
    return (global_min, min_S, min_triple)
global_min, min_S, triple = compute_global_min(200)
print((global_min, min_S, triple))

### Turn 16
def global_min_ratio(limit):
    best = 1e9
    best_trip = None
    for a in range(1, limit+1):
        for b in range(a+1, limit+1):
            for c in range(b+1, limit+1):
                ratio = math.lcm(a,b,c) / (a+b+c)
                if ratio < best:
                    best = ratio
                    best_trip = (a,b,c)
    return best, best_trip

best,trip = global_min_ratio(100)
best,trip


### Turn 16
def global_min_ratio(limit):
    best = 1000000000.0
    best_trip = None
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):
            for c in range(b + 1, limit + 1):
                ratio = math.lcm(a, b, c) / (a + b + c)
                if ratio < best:
                    best = ratio
                    best_trip = (a, b, c)
    return (best, best_trip)
best, trip = global_min_ratio(100)
print((best, trip))

### Turn 17
pow(3, 2025, 11)  # compute 3^2025 mod 11 (but exponent huge maybe not needed)


### Turn 17
print(pow(3, 2025, 11))

### Turn 18
def order_mod(a, mod):
    cur = 1
    for i in range(1, mod):
        cur = (cur*a) % mod
        if cur == 1:
            return i
    return None
order_mod(3, 11)


### Turn 18
def order_mod(a, mod):
    cur = 1
    for i in range(1, mod):
        cur = cur * a % mod
        if cur == 1:
            return i
    return None
print(order_mod(3, 11))

### Turn 19
c_list = [0, 4*pow(3,2025,10**9)]  # that's not needed; we have c values defined

c_vals = [0, 4*3**0, 1848374, 10162574, 265710644, 44636594]  # Actually 4M not needed mod 11, we compute as 4*M mod 11 = 4*1=4
c_vals = [0, 4, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_vals]
mods


### Turn 19
c_list = [0, 4 * pow(3, 2025, 10 ** 9)]
c_vals = [0, 4 * 3 ** 0, 1848374, 10162574, 265710644, 44636594]
c_vals = [0, 4, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_vals]
print(mods)

### Turn 20
c_vals_small = [0, 4*pow(3,0), 1848374, 10162574, 265710644, 44636594]  # note 4*3^0=4, not 4M; but actual c=4M is huge, need mod9 only
mods9 = [0%9, 0%9, 1848374%9, 10162574%9, 265710644%9, 44636594%9]
mods9


### Turn 20
c_vals_small = [0, 4 * pow(3, 0), 1848374, 10162574, 265710644, 44636594]
mods9 = [0 % 9, 0 % 9, 1848374 % 9, 10162574 % 9, 265710644 % 9, 44636594 % 9]
print(mods9)

### Turn 21
def mod_small(c, p):
    if p == 3:
        return c % p
    else:
        return (1 + c) % p  # since M ≡1 mod p

c_vals = [0, None, 1848374, 10162574, 265710644, 44636594]  # we will handle c=4M separately
# For c=4M, n=5M, not using c+1 formula.
c_vals_small = [0, 1848374, 10162574, 265710644, 44636594]
for p in [5,7,11,13,17,19]:
    print("p",p, [mod_small(c, p) for c in c_vals_small])


### Turn 21
def mod_small(c, p):
    if p == 3:
        return c % p
    else:
        return (1 + c) % p
c_vals = [0, None, 1848374, 10162574, 265710644, 44636594]
c_vals_small = [0, 1848374, 10162574, 265710644, 44636594]
for p in [5, 7, 11, 13, 17, 19]:
    print('p', p, [mod_small(c, p) for c in c_vals_small])

### Turn 22
def v3(x):
    cnt=0
    while x%3==0 and x>0:
        x//=3
        cnt+=1
    return cnt

c_vals = [0, 1848374, 10162574, 265710644, 44636594]
[(c, v3(c)) for c in c_vals]


### Turn 22
def v3(x):
    cnt = 0
    while x % 3 == 0 and x > 0:
        x //= 3
        cnt += 1
    return cnt
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
print([(c, v3(c)) for c in c_vals])

### Turn 23
import sympy as sp
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    n = c+1
    print(c, sp.factorint(n))


### Turn 23
import sympy as sp
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    n = c + 1
    print(c, sp.factorint(n))

### Turn 24
[(c, c%3) for c in [1848374,10162574,265710644,44636594]]


### Turn 24
print([(c, c % 3) for c in [1848374, 10162574, 265710644, 44636594]])

### Turn 25
# Precompute r_min(S) for S up to, say, 500
r_min = {}
for S in range(6, 501):
    r,_ = r_min_S(S)
    r_min[S] = r

# Find S with low r values
sorted_r = sorted([(r_min[S], S) for S in r_min])
sorted_r[:20]


### Turn 25
r_min = {}
for S in range(6, 501):
    r, _ = r_min_S(S)
    r_min[S] = r
sorted_r = sorted([(r_min[S], S) for S in r_min])
print(head(sorted_r, 20))

### Turn 26
# Get next distinct ratio after 6/11
ratios = {}
for S,r in r_min.items():
    ratios.setdefault(r, []).append(S)
sorted_ratios = sorted(ratios.keys())
sorted_ratios[:10]


### Turn 26
ratios = {}
for S, r in r_min.items():
    ratios.setdefault(r, []).append(S)
sorted_ratios = sorted(ratios.keys())
print(head(sorted_ratios, 10))

### Turn 27
ratios[0.5714285714285714][:10]


### Turn 27
print(head(ratios[0.5714285714285714], 10))

### Turn 28
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, c%7)


### Turn 28
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, c % 7)

### Turn 29
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, c%5)


### Turn 29
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, c % 5)

### Turn 30
r_min[97]


### Turn 30
print(r_min[97])

### Turn 31
def f_brut_n(n):
    return f_brut_small(n)  # works for moderate n

M_test = 3**6
n_test = 5 * M_test
print(M_test, n_test)
print(f_brut_n(n_test))


### Turn 31
def f_brut_n(n):
    return f_brut_small(n)
M_test = 3 ** 6
n_test = 5 * M_test
print(M_test, n_test)
print(f_brut_n(n_test))

### Turn 32
import fractions, math
fra = fractions.Fraction(2430,3645)
fra


### Turn 32
import fractions, math
fra = fractions.Fraction(2430, 3645)
print(fra)

### Turn 33
def min_ratio_for_c(c):
    # compute n = M + c, but for small c case we only need divisor info from c+1.
    # Determine small S dividing n (>=6). For c where c%3 !=0, n not divisible by 3.
    # So S must be divisor of c+1 not divisible by 3.
    # Compute all divisors of c+1.
    n_cplus1 = c+1
    fac = sp.factorint(n_cplus1)
    # generate divisors
    divisors = [1]
    for p, e in fac.items():
        new_div = []
        for d in divisors:
            for exp in range(1, e+1):
                new_div.append(d * (p**exp))
        divisors += new_div
    # filter
    valid_S = [d for d in set(divisors) if d >= 6 and d % 2 == 1]  # odd
    # Exclude multiples of 3 (since n not multiple of 3)
    valid_S = [d for d in valid_S if d % 3 != 0]
    # For each S compute r_min[S] precomputed up to some limit; but S may exceed 500?
    # We'll compute r_min(S) on the fly via enumeration of triples up to S maybe heavy.
    # But we can compute r_min for S up to maybe 1000 using earlier method.
    return valid_S

for c in [1848374,10162574,265710644,44636594]:
    Slist = min_ratio_for_c(c)
    print(c, len(Slist), sorted(Slist)[:10])


### Turn 33
def min_ratio_for_c(c):
    n_cplus1 = c + 1
    fac = sp.factorint(n_cplus1)
    divisors = [1]
    for p, e in fac.items():
        new_div = []
        for d in divisors:
            for exp in range(1, e + 1):
                new_div.append(d * p ** exp)
        divisors += new_div
    valid_S = [d for d in set(divisors) if d >= 6 and d % 2 == 1]
    valid_S = [d for d in valid_S if d % 3 != 0]
    return valid_S
for c in [1848374, 10162574, 265710644, 44636594]:
    Slist = min_ratio_for_c(c)
    print(c, len(Slist), head(sorted(Slist), 10))

### Turn 34
def ratio_S_mod1(S):
    return (2/3)*(1 - 1/S)
print(ratio_S_mod1(775))


### Turn 34
def ratio_S_mod1(S):
    return 2 / 3 * (1 - 1 / S)
print(ratio_S_mod1(775))

### Turn 35
def approx_rmin(S):
    # candidate formulas
    cand = []
    if S % 11 == 0:
        cand.append(6/11)
    if S % 7 == 0:
        cand.append(4/7)
    if S % 3 == 1:
        cand.append((2/3)*(1 - 1/S))
    # also base case: maybe use triple (1, (S-1)/2, (S-1)/2) not allowed due to duplicate
    # So fallback to actual r_min
    actual = r_min[S]
    cand.append(actual)
    return min(cand), actual

diffs = []
for S in range(6, 501):
    approx, actual = approx_rmin(S)
    if abs(approx - actual) > 1e-12:
        diffs.append((S, approx, actual))
len(diffs)


### Turn 35
def approx_rmin(S):
    cand = []
    if S % 11 == 0:
        cand.append(6 / 11)
    if S % 7 == 0:
        cand.append(4 / 7)
    if S % 3 == 1:
        cand.append(2 / 3 * (1 - 1 / S))
    actual = r_min[S]
    cand.append(actual)
    return (min(cand), actual)
diffs = []
for S in range(6, 501):
    approx, actual = approx_rmin(S)
    if abs(approx - actual) > 1e-12:
        diffs.append((S, approx, actual))
print(len(diffs))

### Turn 36
def min_ratio_for_n(c):
    # compute n = M + c
    # Since we only need small divisors derived from c+1 (for c%3!=0)
    n_mod3 = c % 3
    if n_mod3 == 0:
        # n divisible by 3, so ratio = 2/3 (via divisor 9)
        return (2/3)
    # else n not multiple of 3
    # compute factorization of c+1, get divisors not divisible by 3
    fac = sp.factorint(c+1)
    # generate odd divisors (excluding factor 3)
    # remove factor 3 from fac
    fac_n3 = {p:e for p,e in fac.items() if p != 3}
    # generate divisors
    divisors = [1]
    for p, e in fac_n3.items():
        new = []
        for d in divisors:
            for exp in range(1, e+1):
                new.append(d * (p**exp))
        divisors += new
    divisors = set(divisors)
    # exclude 1,2,3,4,5 maybe
    valid_S = [d for d in divisors if d >= 6]
    # Now find minimal ratio according to rules
    best_R = None
    best_S = None
    for S in valid_S:
        # compute ratio based on categories
        if S % 11 == 0:
            R = 6/11
        elif S % 7 == 0:
            R = 4/7
        elif S % 9 == 0:
            # not possible since n not multiple of 3 => S not divisible by 3
            R = 2/3
        elif S % 3 == 1:
            R = (2/3)*(1 - 1/S)
        else:
            # fallback compute actual r_min for S via brute (could be bigger)
            R = r_min[S] if S in r_min else None
        if best_R is None or R < best_R:
            best_R = R
            best_S = S
    return best_R, best_S

for c in [1848374,10162574,265710644,44636594]:
    R,S = min_ratio_for_n(c)
    print(c, "R=",R, "S=",S)


### Turn 36
def min_ratio_for_n(c):
    n_mod3 = c % 3
    if n_mod3 == 0:
        return 2 / 3
    fac = sp.factorint(c + 1)
    fac_n3 = {p: e for p, e in fac.items() if p != 3}
    divisors = [1]
    for p, e in fac_n3.items():
        new = []
        for d in divisors:
            for exp in range(1, e + 1):
                new.append(d * p ** exp)
        divisors += new
    divisors = set(divisors)
    valid_S = [d for d in divisors if d >= 6]
    best_R = None
    best_S = None
    for S in valid_S:
        if S % 11 == 0:
            R = 6 / 11
        elif S % 7 == 0:
            R = 4 / 7
        elif S % 9 == 0:
            R = 2 / 3
        elif S % 3 == 1:
            R = 2 / 3 * (1 - 1 / S)
        else:
            R = r_min[S] if S in r_min else None
        if best_R is None or R < best_R:
            best_R = R
            best_S = S
    return (best_R, best_S)
for c in [1848374, 10162574, 265710644, 44636594]:
    R, S = min_ratio_for_n(c)
    print(c, 'R=', R, 'S=', S)

### Turn 37
def r_min_exact(S):
    # compute via earlier function r_min_S (brute)
    return r_min_S(S)[0]

mod_data = {}
for S in range(6,201):
    r = r_min_exact(S)
    mod = S % 3
    mod_data.setdefault(mod, []).append((S,r))
# Show some values
for mod in [0,1,2]:
    print("mod",mod, "examples:", mod_data[mod][:10])


### Turn 37
def r_min_exact(S):
    return r_min_S(S)[0]
mod_data = {}
for S in range(6, 201):
    r = r_min_exact(S)
    mod = S % 3
    mod_data.setdefault(mod, []).append((S, r))
for mod in [0, 1, 2]:
    print('mod', mod, 'examples:', head(mod_data[mod], 10))

### Turn 38
def best_ratio_for_n(c):
    # compute best ratio for n = M + c
    # M huge but properties as earlier
    # Determine if n divisible by 11 or 7 or 9
    # n mod p = (M mod p + c mod p) where M mod p = 1 for p !=3; for p=3 M mod 3 =0
    # check divisibility
    # compute n_mod11:
    def n_mod(p):
        if p == 3:
            return (0 + c) % p
        else:
            return (1 + c % p) % p
    # Check 11
    if n_mod(11) == 0:
        return (6/11)
    if n_mod(7) == 0:
        return (4/7)
    if n_mod(9) == 0:
        # need n divisible by 9 (i.e., c multiple of 9) because M divisible by 9
        # n_mod(9) computed with M%9 = 0; so n_mod(9) = c %9
        if c % 9 == 0:
            return (2/3)
    # else find smallest divisor S (>=6) of n with S%3==1
    # Since n's odd divisors are divisors of c+1 (excluding factor 3)
    # We'll generate divisors from factorization of c+1.
    fac = sp.factorint(c+1)
    # remove prime 3 if present
    fac = {p:e for p,e in fac.items() if p != 3}
    # generate divisors
    divisors = [1]
    for p,e in fac.items():
        new = []
        for d in divisors:
            for exp in range(1, e+1):
                new.append(d * (p**exp))
        divisors += new
    divisors = set(divisors)
    valid = [d for d in divisors if d >= 6 and d % 3 == 1]
    if not valid:
        # fallback: use actual r_min for each divisor maybe (rare)
        # compute min r via brute for each divisor up to maybe 1000 else return something else.
        # For safety compute actual ratio via r_min_S for each divisor (maybe large)
        best_R = None
        for d in divisors:
            if d >=6:
                r = r_min_S(d)[0]
                if best_R is None or r < best_R:
                    best_R = r
        return best_R
    S = min(valid)
    R = (2/3)*(1 - 1/S)
    return R, S

for c in [1848374,10162574,265710644,44636594]:
    val = best_ratio_for_n(c)
    print(c, val)


### Turn 38
def best_ratio_for_n(c):

    def n_mod(p):
        if p == 3:
            return (0 + c) % p
        else:
            return (1 + c % p) % p
    if n_mod(11) == 0:
        return 6 / 11
    if n_mod(7) == 0:
        return 4 / 7
    if n_mod(9) == 0:
        if c % 9 == 0:
            return 2 / 3
    fac = sp.factorint(c + 1)
    fac = {p: e for p, e in fac.items() if p != 3}
    divisors = [1]
    for p, e in fac.items():
        new = []
        for d in divisors:
            for exp in range(1, e + 1):
                new.append(d * p ** exp)
        divisors += new
    divisors = set(divisors)
    valid = [d for d in divisors if d >= 6 and d % 3 == 1]
    if not valid:
        best_R = None
        for d in divisors:
            if d >= 6:
                r = r_min_S(d)[0]
                if best_R is None or r < best_R:
                    best_R = r
        return best_R
    S = min(valid)
    R = 2 / 3 * (1 - 1 / S)
    return (R, S)
for c in [1848374, 10162574, 265710644, 44636594]:
    val = best_ratio_for_n(c)
    print(c, val)

### Turn 39
import math
math.gcd(192,291)


### Turn 39
import math
print(math.gcd(192, 291))

### Turn 40
math.gcd(204,309)


### Turn 40
print(math.gcd(204, 309))

### Turn 41
from fractions import Fraction

g0 = Fraction(2,3)
g4M = Fraction(10,3)
g1 = Fraction(16,25)
g2 = Fraction(16,25)
g3 = Fraction(64,97)
g4 = Fraction(68,103)

total = g0 + g4M + g1 + g2 + g3 + g4
total


### Turn 41
from fractions import Fraction
g0 = Fraction(2, 3)
g4M = Fraction(10, 3)
g1 = Fraction(16, 25)
g2 = Fraction(16, 25)
g3 = Fraction(64, 97)
g4 = Fraction(68, 103)
total = g0 + g4M + g1 + g2 + g3 + g4
print(total)

### Turn 42
import math
math.gcd(1648512, 249775)


### Turn 42
import math
print(math.gcd(1648512, 249775))

### Turn 43
(p+q) % 99991


### Turn 43
print((p + q) % 99991)

### Turn 44
p = 1648512
q = 249775
(p+q) % 99991


### Turn 44
p = 1648512
q = 249775
print((p + q) % 99991)

### Turn 45
def n_mod_p(c,p):
    if p == 3:
        return (0 + c)%p
    else:
        return (1 + c%p)%p

c_list = [0, 4*pow(3,2025,11) % 11, 1848374, 10162574, 265710644, 44636594]
# Actually 4M mod 11 = 4*1=4, so n_mod = 1+4 =5 mod11 => not zero.
c_values = [0, None, 1848374, 10162574, 265710644, 44636594]
for c in [1848374,10162574,265710644,44636594]:
    print(c, n_mod_p(c, 11), n_mod_p(c,7), n_mod_p(c,9))


### Turn 45
def n_mod_p(c, p):
    if p == 3:
        return (0 + c) % p
    else:
        return (1 + c % p) % p
c_list = [0, 4 * pow(3, 2025, 11) % 11, 1848374, 10162574, 265710644, 44636594]
c_values = [0, None, 1848374, 10162574, 265710644, 44636594]
for c in [1848374, 10162574, 265710644, 44636594]:
    print(c, n_mod_p(c, 11), n_mod_p(c, 7), n_mod_p(c, 9))

### Turn 46
from fractions import Fraction
g_vals = [Fraction(2,3), Fraction(10,3), Fraction(16,25), Fraction(16,25), Fraction(64,97), Fraction(68,103)]
total = sum(g_vals, Fraction(0,1))
total, total.numerator, total.denominator


### Turn 46
from fractions import Fraction
g_vals = [Fraction(2, 3), Fraction(10, 3), Fraction(16, 25), Fraction(16, 25), Fraction(64, 97), Fraction(68, 103)]
total = sum(g_vals, Fraction(0, 1))
print((total, total.numerator, total.denominator))

### Turn 47
(p+q) % 99991


### Turn 47
print((p + q) % 99991)

