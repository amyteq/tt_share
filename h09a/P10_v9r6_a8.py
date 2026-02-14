# Auto-extracted from: P10_v9r6.md
# Attempt: 8

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
def lcm(a,b):
    return a*b//math.gcd(a,b)
def lcm3(a,b,c):
    return lcm(lcm(a,b),c)
best = (1e9,None)
ratios = {}
for x in range(1,50):
    for y in range(x+1,50):
        for z in range(y+1,50):
            s = x+y+z
            L = lcm3(x,y,z)
            ratio = L/s
            if ratio < best[0]:
                best = (ratio, (x,y,z))
            # store minimal per sum?
            if s not in ratios or ratio < ratios[s][0]:
                ratios[s] = (ratio,(x,y,z))
best, len(ratios)


### Turn 0
import math, itertools

def lcm(a, b):
    return a * b // math.gcd(a, b)

def lcm3(a, b, c):
    return lcm(lcm(a, b), c)
best = (1000000000.0, None)
ratios = {}
for x in range(1, 50):
    for y in range(x + 1, 50):
        for z in range(y + 1, 50):
            s = x + y + z
            L = lcm3(x, y, z)
            ratio = L / s
            if ratio < best[0]:
                best = (ratio, (x, y, z))
            if s not in ratios or ratio < ratios[s][0]:
                ratios[s] = (ratio, (x, y, z))
print((best, len(ratios)))

### Turn 1
best = (1e9,None)
for x in range(1,101):
    for y in range(x+1,101):
        for z in range(y+1,101):
            s = x+y+z
            L = lcm3(x,y,z)
            ratio = L/s
            if ratio < best[0]:
                best = (ratio,(x,y,z))
best


### Turn 1
best = (1000000000.0, None)
for x in range(1, 101):
    for y in range(x + 1, 101):
        for z in range(y + 1, 101):
            s = x + y + z
            L = lcm3(x, y, z)
            ratio = L / s
            if ratio < best[0]:
                best = (ratio, (x, y, z))
print(best)

### Turn 2
best = (1, None, None)
for p in range(2,50):
    for q in range(p+1,50):
        if math.gcd(p,q)==1:
            ratio = p*q/(p+q+p*q)
            if ratio < best[0]:
                best = (ratio, p, q)
best


### Turn 2
best = (1, None, None)
for p in range(2, 50):
    for q in range(p + 1, 50):
        if math.gcd(p, q) == 1:
            ratio = p * q / (p + q + p * q)
            if ratio < best[0]:
                best = (ratio, p, q)
print(best)

### Turn 3
import math, itertools, fractions, sys, collections, json, math
def min_lcm_for_sum(S):
    min_L = None
    min_trip = None
    # iterate over a<b<c, distinct positive ints summing to S
    for a in range(1, S-1):
        for b in range(a+1, S-a):
            c = S - a - b
            if c <= b:
                continue
            L = math.lcm(a,b,c)
            if min_L is None or L < min_L:
                min_L = L
                min_trip = (a,b,c)
    return min_L, min_trip

maxS = 200
Lmin = {}
ratio = {}
triplet = {}
for S in range(6, maxS+1):
    L, t = min_lcm_for_sum(S)
    Lmin[S] = L
    triplet[S] = t
    ratio[S] = fractions.Fraction(L, S)
len(Lmin)


### Turn 3
import math, itertools, fractions, sys, collections, json, math

def min_lcm_for_sum(S):
    min_L = None
    min_trip = None
    for a in range(1, S - 1):
        for b in range(a + 1, S - a):
            c = S - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if min_L is None or L < min_L:
                min_L = L
                min_trip = (a, b, c)
    return (min_L, min_trip)
maxS = 200
Lmin = {}
ratio = {}
triplet = {}
for S in range(6, maxS + 1):
    L, t = min_lcm_for_sum(S)
    Lmin[S] = L
    triplet[S] = t
    ratio[S] = fractions.Fraction(L, S)
print(len(Lmin))

### Turn 4
sorted_ratios = sorted(((ratio[S], S, Lmin[S], triplet[S]) for S in ratio), key=lambda x: x[0])
sorted_ratios[:20]


### Turn 4
sorted_ratios = sorted(((ratio[S], S, Lmin[S], triplet[S]) for S in ratio), key=lambda x: x[0])
print(head(sorted_ratios, 20))

### Turn 5
# compute minimal ratio for each S
min_ratios = {}
for S in range(6,101):
    min_ratios[S] = ratio[S]
sorted((v,S) for S,v in min_ratios.items())[:20]


### Turn 5
min_ratios = {}
for S in range(6, 101):
    min_ratios[S] = ratio[S]
print(head(sorted(((v, S) for S, v in min_ratios.items())), 20))

### Turn 6
sorted_ratios_small = sorted(((ratio[S], S) for S in ratio), key=lambda x: x[0])
sorted_ratios_small[:30]


### Turn 6
sorted_ratios_small = sorted(((ratio[S], S) for S in ratio), key=lambda x: x[0])
print(head(sorted_ratios_small, 30))

### Turn 7
sorted_ratios_small[30:60]


### Turn 7
print(head(sorted_ratios_small, 60))

### Turn 8
ratio[10], Lmin[10], triplet[10]


### Turn 8
print((ratio[10], Lmin[10], triplet[10]))

### Turn 9
# find minimal ratio for each S
ratios_set = {}
for S in range(6,200):
    ratios_set[ratio[S]] = ratios_set.get(ratio[S], []) + [S]
sorted_ratio_values = sorted(ratios_set.keys())
sorted_ratio_values[:20]


### Turn 9
ratios_set = {}
for S in range(6, 200):
    ratios_set[ratio[S]] = ratios_set.get(ratio[S], []) + [S]
sorted_ratio_values = sorted(ratios_set.keys())
print(head(sorted_ratio_values, 20))

### Turn 10
ratio[27], Lmin[27], triplet[27]


### Turn 10
print((ratio[27], Lmin[27], triplet[27]))

### Turn 11
def compute_ratio_for_S(S):
    L,_ = min_lcm_for_sum(S)
    return fractions.Fraction(L, S)
for a in range(2,9):
    S = 3**a
    print(S, compute_ratio_for_S(S))


### Turn 11
def compute_ratio_for_S(S):
    L, _ = min_lcm_for_sum(S)
    return fractions.Fraction(L, S)
for a in range(2, 9):
    S = 3 ** a
    print(S, compute_ratio_for_S(S))

### Turn 12
ratio[15], Lmin[15], triplet[15]


### Turn 12
print((ratio[15], Lmin[15], triplet[15]))

### Turn 13
Lmin[13], triplet[13], ratio[13]


### Turn 13
print((Lmin[13], triplet[13], ratio[13]))

### Turn 14
def ratio_for_S(S):
    L,_ = min_lcm_for_sum(S)
    return fractions.Fraction(L, S)
ratios_primes = {}
for p in range(6,101):
    if sympy.isprime(p):
        ratios_primes[p] = ratio_for_S(p)
ratios_primes


### Turn 14
def ratio_for_S(S):
    L, _ = min_lcm_for_sum(S)
    return fractions.Fraction(L, S)
ratios_primes = {}
for p in range(6, 101):
    if sympy.isprime(p):
        ratios_primes[p] = ratio_for_S(p)
print(ratios_primes)

### Turn 15
ratio[6], Lmin[6], triplet[6]


### Turn 15
print((ratio[6], Lmin[6], triplet[6]))

### Turn 16
import sympy as sp, math, fractions, itertools, sys, math, random, functools, collections, json, typing, math
# compute minimal L for sums up to B
B = 200
Lmin_small = {}
ratio_small = {}
triplet_small = {}
for S in range(6, B+1):
    # compute minimal L for S using brute force enumeration a<b<c
    minL = None
    mintrip = None
    for a in range(1, S-1):
        for b in range(a+1, S-a):
            c = S - a - b
            if c <= b:
                continue
            L = math.lcm(a,b,c)
            if minL is None or L < minL:
                minL = L
                mintrip = (a,b,c)
    Lmin_small[S] = minL
    ratio_small[S] = fractions.Fraction(minL, S)
    triplet_small[S] = mintrip

# sort candidates by ratio ascending, then by S to ensure deterministic
candidates = sorted([(ratio_small[S], S) for S in ratio_small], key=lambda x: (x[0], x[1]))
candidates[:10]


### Turn 16
import sympy as sp, math, fractions, itertools, sys, math, random, functools, collections, json, typing, math
B = 200
Lmin_small = {}
ratio_small = {}
triplet_small = {}
for S in range(6, B + 1):
    minL = None
    mintrip = None
    for a in range(1, S - 1):
        for b in range(a + 1, S - a):
            c = S - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if minL is None or L < minL:
                minL = L
                mintrip = (a, b, c)
    Lmin_small[S] = minL
    ratio_small[S] = fractions.Fraction(minL, S)
    triplet_small[S] = mintrip
candidates = sorted([(ratio_small[S], S) for S in ratio_small], key=lambda x: (x[0], x[1]))
print(head(candidates, 10))

### Turn 17
# show first 30 candidates
for i, (r,s) in enumerate(candidates[:30]):
    print(i, s, r)


### Turn 17
for i, (r, s) in enumerate(head(candidates, 30)):
    print(i, s, r)

### Turn 18
for i in range(30, 50):
    r,s = candidates[i]
    print(i, s, r)


### Turn 18
for i in range(30, 50):
    r, s = candidates[i]
    print(i, s, r)

### Turn 19
for i in range(50, 70):
    r,s = candidates[i]
    print(i, s, r)


### Turn 19
for i in range(50, 70):
    r, s = candidates[i]
    print(i, s, r)

### Turn 20
import math, fractions, sympy as sp, itertools, sys, functools, collections, math, random, json, typing, numpy as np
fac = math.factorial(2025)  # big integer
# precomputed candidate list
candidates = sorted([(ratio_small[S], S) for S in ratio_small], key=lambda x: (x[0], x[1]))
# precompute Lmin for each S
L_by_S = Lmin_small

# c values as described
c_values = [0, None, 1848374, 10162574, 265710644, 44636594]
# c = 4M: we will handle specially

# compute M_mod_S for each S in candidates
def factorial_mod(phi):
    # compute 2025! mod phi
    if phi == 1:
        return 0
    res = 1
    for i in range(2, 2026):
        res = (res * i) % phi
    return res

M_mod_by_S = {}
for (r,S) in candidates:
    # compute M mod S
    factors = sp.factorint(S)
    mods = []
    rems = []
    for p, k in factors.items():
        mod_pk = p**k
        if p == 3:
            # Since exponent >= k, remainder is 0
            rem = 0
        else:
            phi = mod_pk - mod_pk // p
            exp_mod = factorial_mod(phi)
            rem = pow(3, exp_mod, mod_pk)
        mods.append(mod_pk)
        rems.append(rem)
    # combine via CRT
    # CRT may return None if inconsistent; but should be consistent.
    crt_res = sp.crt(mods, rems)
    if crt_res[0] is None:
        # if some mod has remainder 0 but other? Should not happen.
        M_mod = 0
    else:
        M_mod = crt_res[0] % sp.ilcm(*mods)  # combined modulus is product because mods are coprime? They are powers of distinct primes, thus lcm = product.
    M_mod_by_S[S] = M_mod

def find_min_D_for_n(M_mod_S, c):
    # if c is integer small (c type int) we treat as integer.
    # returns D and associated L
    for ratio_val, S in candidates:
        mod = M_mod_S[S]  # M mod S
        # compute n_mod = (M_mod_S + c) % S, where for c = '4M', we treat specially (c is None for that case)
        if c == '4M':
            n_mod = (5 * mod) % S   # n = M + 4M =5M
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            # divisor found
            return S, L_by_S[S]
    # fallback: no divisor found within candidates; return S = something? We fallback to D = n (ratio=1)
    # For ratio=1 we need L = n (since we can take 1,2,n-3). But we can't compute L due to huge n.
    # In our dataset, we likely always find a divisor.
    raise ValueError("No divisor found within bound")
    
# Test for each c
c_labels = [0, '4M', 1848374, 10162574, 265710644, 44636594]
for c in c_labels:
    D, L = find_min_D_for_n(M_mod_by_S, c)
    print(c, D, L, ratio_small[D])


### Turn 20
import math, fractions, sympy as sp, itertools, sys, functools, collections, math, random, json, typing, numpy as np
fac = math.factorial(2025)
candidates = sorted([(ratio_small[S], S) for S in ratio_small], key=lambda x: (x[0], x[1]))
L_by_S = Lmin_small
c_values = [0, None, 1848374, 10162574, 265710644, 44636594]

def factorial_mod(phi):
    if phi == 1:
        return 0
    res = 1
    for i in range(2, 2026):
        res = res * i % phi
    return res
M_mod_by_S = {}
for r, S in candidates:
    factors = sp.factorint(S)
    mods = []
    rems = []
    for p, k in factors.items():
        mod_pk = p ** k
        if p == 3:
            rem = 0
        else:
            phi = mod_pk - mod_pk // p
            exp_mod = factorial_mod(phi)
            rem = pow(3, exp_mod, mod_pk)
        mods.append(mod_pk)
        rems.append(rem)
    crt_res = sp.crt(mods, rems)
    if crt_res[0] is None:
        M_mod = 0
    else:
        M_mod = crt_res[0] % sp.ilcm(*mods)
    M_mod_by_S[S] = M_mod

def find_min_D_for_n(M_mod_S, c):
    for ratio_val, S in candidates:
        mod = M_mod_S[S]
        if c == '4M':
            n_mod = 5 * mod % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            return (S, L_by_S[S])
    raise ValueError('No divisor found within bound')
c_labels = [0, '4M', 1848374, 10162574, 265710644, 44636594]
for c in c_labels:
    D, L = find_min_D_for_n(M_mod_by_S, c)
    print(c, D, L, ratio_small[D])

### Turn 21
M_mod_by_S = {}
for ratio_val, S in candidates:
    factors = sp.factorint(S)
    mods = []
    rems = []
    for p, k in factors.items():
        mod_pk = p**k
        if p == 3:
            rem = 0
        else:
            phi = mod_pk - mod_pk // p
            exp_mod = factorial_mod(phi)
            rem = pow(3, exp_mod, mod_pk)
        mods.append(mod_pk)
        rems.append(rem)
    if len(mods) == 1:
        M_mod = rems[0] % mods[0]
    else:
        crt_res = sp.crt(mods, rems)
        M_mod = crt_res[0]  # this is modulo product
    M_mod_by_S[S] = M_mod

def find_min_D_for_n(c):
    for ratio_val, S in candidates:
        mod = M_mod_by_S[S]
        if c == '4M':
            n_mod = (5 * mod) % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            return S, L_by_S[S]
    raise ValueError("No divisor found")

c_labels = [0, '4M', 1848374, 10162574, 265710644, 44636594]
for c in c_labels:
    D,L = find_min_D_for_n(c)
    print(c, D, L, ratio_small[D])


### Turn 21
M_mod_by_S = {}
for ratio_val, S in candidates:
    factors = sp.factorint(S)
    mods = []
    rems = []
    for p, k in factors.items():
        mod_pk = p ** k
        if p == 3:
            rem = 0
        else:
            phi = mod_pk - mod_pk // p
            exp_mod = factorial_mod(phi)
            rem = pow(3, exp_mod, mod_pk)
        mods.append(mod_pk)
        rems.append(rem)
    if len(mods) == 1:
        M_mod = rems[0] % mods[0]
    else:
        crt_res = sp.crt(mods, rems)
        M_mod = crt_res[0]
    M_mod_by_S[S] = M_mod

def find_min_D_for_n(c):
    for ratio_val, S in candidates:
        mod = M_mod_by_S[S]
        if c == '4M':
            n_mod = 5 * mod % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            return (S, L_by_S[S])
    raise ValueError('No divisor found')
c_labels = [0, '4M', 1848374, 10162574, 265710644, 44636594]
for c in c_labels:
    D, L = find_min_D_for_n(c)
    print(c, D, L, ratio_small[D])

### Turn 22
def compute_A(D, L):
    return (fac * L) // D  # floor division
# compute for each c
c_list = [0, '4M', 1848374, 10162574, 265710644, 44636594]
A_list = []
for c in c_list:
    D, L = find_min_D_for_n(c)
    A = compute_A(D, L)
    A_list.append((c, D, L, A))
    print(c, D, L, "A (fac*L//D) bits", A.bit_length())


### Turn 22
def compute_A(D, L):
    return fac * L // D
c_list = [0, '4M', 1848374, 10162574, 265710644, 44636594]
A_list = []
for c in c_list:
    D, L = find_min_D_for_n(c)
    A = compute_A(D, L)
    A_list.append((c, D, L, A))
    print(c, D, L, 'A (fac*L//D) bits', A.bit_length())

### Turn 23
from fractions import Fraction
total_A = sum(A for _,_,_,A in A_list)
total_frac = Fraction(total_A, fac)  # automatically reduces
p = total_frac.numerator
q = total_frac.denominator
p_mod = p % 99991
q_mod = q % 99991
(p_mod, q_mod, (p+q) % 99991)


### Turn 23
from fractions import Fraction
total_A = sum((A for _, _, _, A in A_list))
total_frac = Fraction(total_A, fac)
p = total_frac.numerator
q = total_frac.denominator
p_mod = p % 99991
q_mod = q % 99991
print((p_mod, q_mod, (p + q) % 99991))

### Turn 24
triplet_small[47]


### Turn 24
print(triplet_small[47])

### Turn 25
triplet_small[97]


### Turn 25
print(triplet_small[97])

### Turn 26
triplet_small[167]


### Turn 26
print(triplet_small[167])

### Turn 27
total_frac_numer = total_frac.numerator
total_frac_denom = total_frac.denominator
total_frac_numer, total_frac_denom, math.gcd(total_frac_numer, total_frac_denom)


### Turn 27
total_frac_numer = total_frac.numerator
total_frac_denom = total_frac.denominator
print((total_frac_numer, total_frac_denom, math.gcd(total_frac_numer, total_frac_denom)))

### Turn 28
(p+q) % 99991


### Turn 28
print((p + q) % 99991)

### Turn 29
unique_ratios = {}
for S in range(6, 201):
    unique_ratios[ratio_small[S]] = (S, Lmin_small[S])
sorted_unique = sorted(unique_ratios.items(), key=lambda x: x[0])
sorted_unique[:30]


### Turn 29
unique_ratios = {}
for S in range(6, 201):
    unique_ratios[ratio_small[S]] = (S, Lmin_small[S])
sorted_unique = sorted(unique_ratios.items(), key=lambda x: x[0])
print(head(sorted_unique, 30))

### Turn 30
def is_divisible_by_S(c, S):
    M_mod = M_mod_by_S[S]
    if c == '4M':
        n_mod = (5 * M_mod) % S
    else:
        n_mod = (M_mod + c) % S
    return n_mod == 0

for S in [11,7,17,10,23,13]:
    print("S", S, "divisible:", is_divisible_by_S(1848374, S))


### Turn 30
def is_divisible_by_S(c, S):
    M_mod = M_mod_by_S[S]
    if c == '4M':
        n_mod = 5 * M_mod % S
    else:
        n_mod = (M_mod + c) % S
    return n_mod == 0
for S in [11, 7, 17, 10, 23, 13]:
    print('S', S, 'divisible:', is_divisible_by_S(1848374, S))

### Turn 31
def compute_Lmin_up_to(N):
    Lmin = {}
    for S in range(6, N+1):
        minL = None
        mintrip = None
        # iterate a from 1 to S//3
        for a in range(1, S//3 + 1):
            for b in range(a+1, (S - a)//2 + 1):
                c = S - a - b
                if c <= b:
                    continue
                L = math.lcm(a,b,c)
                if minL is None or L < minL:
                    minL = L
                    mintrip = (a,b,c)
        Lmin[S] = (minL, mintrip)
    return Lmin

Lmin_300 = compute_Lmin_up_to(300)
# compute minimal ratio among divisors for each S as earlier
ratio_300 = {}
for S in range(6, 301):
    # minimal ratio over divisors of S using Lmin values for divisors
    # Actually we already have ratio for each S via minimal L directly computed
    minL,_ = Lmin_300[S]
    ratio_300[S] = fractions.Fraction(minL, S)
    
# find minimal ratio overall up to 300
min_ratio_overall = min(ratio_300.values())
min_ratio_overall


### Turn 31
def compute_Lmin_up_to(N):
    Lmin = {}
    for S in range(6, N + 1):
        minL = None
        mintrip = None
        for a in range(1, S // 3 + 1):
            for b in range(a + 1, (S - a) // 2 + 1):
                c = S - a - b
                if c <= b:
                    continue
                L = math.lcm(a, b, c)
                if minL is None or L < minL:
                    minL = L
                    mintrip = (a, b, c)
        Lmin[S] = (minL, mintrip)
    return Lmin
Lmin_300 = compute_Lmin_up_to(300)
ratio_300 = {}
for S in range(6, 301):
    minL, _ = Lmin_300[S]
    ratio_300[S] = fractions.Fraction(minL, S)
min_ratio_overall = min(ratio_300.values())
print(min_ratio_overall)

### Turn 32
sorted([(ratio_300[S], S) for S in range(6,301)], key=lambda x: (x[0], x[1]))[:20]


### Turn 32
print(head(sorted([(ratio_300[S], S) for S in range(6, 301)], key=lambda x: (x[0], x[1])), 20))

### Turn 33
sorted([(ratio_300[S], S) for S in range(6,301)], key=lambda x: (x[0], x[1]))[20:40]


### Turn 33
print(head(sorted([(ratio_300[S], S) for S in range(6, 301)], key=lambda x: (x[0], x[1])), 40))

### Turn 34
def minimal_ratio_for_n(c, max_S=300):
    best_ratio = fractions.Fraction(1,1)  # ratio 1 as worst-case (since L >= S)
    best_S = None
    for S in range(6, max_S+1):
        mod = M_mod_by_S[S]
        if c == '4M':
            n_mod = (5 * mod) % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            # divisor found, compute ratio
            r = ratio_small.get(S)
            if r is None:
                # we haven't precomputed ratio_small for S > B; compute directly using Lmin_300
                L,_ = Lmin_300[S]
                r = fractions.Fraction(L, S)
            if r < best_ratio:
                best_ratio = r
                best_S = S
    return best_S, best_ratio

test_c = [0,'4M',1848374,10162574,265710644,44636594]
for c in test_c:
    best_S, best_r = minimal_ratio_for_n(c, max_S=300)
    print(c, best_S, best_r)


### Turn 34
def minimal_ratio_for_n(c, max_S=300):
    best_ratio = fractions.Fraction(1, 1)
    best_S = None
    for S in range(6, max_S + 1):
        mod = M_mod_by_S[S]
        if c == '4M':
            n_mod = 5 * mod % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            r = ratio_small.get(S)
            if r is None:
                L, _ = Lmin_300[S]
                r = fractions.Fraction(L, S)
            if r < best_ratio:
                best_ratio = r
                best_S = S
    return (best_S, best_ratio)
test_c = [0, '4M', 1848374, 10162574, 265710644, 44636594]
for c in test_c:
    best_S, best_r = minimal_ratio_for_n(c, max_S=300)
    print(c, best_S, best_r)

### Turn 35
def compute_M_mod_S_full(max_S):
    M_mod = {}
    for S in range(6, max_S+1):
        factors = sp.factorint(S)
        mods = []
        rems = []
        for p,k in factors.items():
            mod_pk = p**k
            if p == 3:
                rem = 0
            else:
                phi = mod_pk - mod_pk // p
                exp_mod = factorial_mod(phi)
                rem = pow(3, exp_mod, mod_pk)
            mods.append(mod_pk)
            rems.append(rem)
        if len(mods) == 1:
            M_mod_S = rems[0] % mods[0]
        else:
            M_mod_S = sp.crt(mods, rems)[0]
        M_mod[S] = M_mod_S
    return M_mod

M_mod_all = compute_M_mod_S_full(300)


### Turn 35
def compute_M_mod_S_full(max_S):
    M_mod = {}
    for S in range(6, max_S + 1):
        factors = sp.factorint(S)
        mods = []
        rems = []
        for p, k in factors.items():
            mod_pk = p ** k
            if p == 3:
                rem = 0
            else:
                phi = mod_pk - mod_pk // p
                exp_mod = factorial_mod(phi)
                rem = pow(3, exp_mod, mod_pk)
            mods.append(mod_pk)
            rems.append(rem)
        if len(mods) == 1:
            M_mod_S = rems[0] % mods[0]
        else:
            M_mod_S = sp.crt(mods, rems)[0]
        M_mod[S] = M_mod_S
    return M_mod
print(M_mod_all)

### Turn 36
def minimal_ratio_for_n_full(c, max_S=300):
    best_ratio = fractions.Fraction(1,1)  # start high (1)
    best_S = None
    for S in range(6, max_S+1):
        mod = M_mod_all[S]
        if c == '4M':
            n_mod = (5 * mod) % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            # divisor found
            # compute minimal L for S (using Lmin_300)
            L,_ = Lmin_300[S]
            r = fractions.Fraction(L, S)
            if r < best_ratio:
                best_ratio = r
                best_S = S
    return best_S, best_ratio

for c in test_c:
    best_S, best_r = minimal_ratio_for_n_full(c, max_S=300)
    print(c, best_S, best_r)


### Turn 36
def minimal_ratio_for_n_full(c, max_S=300):
    best_ratio = fractions.Fraction(1, 1)
    best_S = None
    for S in range(6, max_S + 1):
        mod = M_mod_all[S]
        if c == '4M':
            n_mod = 5 * mod % S
        else:
            n_mod = (mod + c) % S
        if n_mod == 0:
            L, _ = Lmin_300[S]
            r = fractions.Fraction(L, S)
            if r < best_ratio:
                best_ratio = r
                best_S = S
    return (best_S, best_ratio)
for c in test_c:
    best_S, best_r = minimal_ratio_for_n_full(c, max_S=300)
    print(c, best_S, best_r)

### Turn 37
def compute_g(c):
    D, L = find_min_D_for_n(c)
    A = (fac * L) // D
    # return Fraction(A, fac)
    return fractions.Fraction(A, fac)

for c in c_labels:
    g_val = compute_g(c)
    print(c, g_val)


### Turn 37
def compute_g(c):
    D, L = find_min_D_for_n(c)
    A = fac * L // D
    return fractions.Fraction(A, fac)
for c in c_labels:
    g_val = compute_g(c)
    print(c, g_val)

### Turn 38
from fractions import Fraction
sum_frac = sum([Fraction(2,3), Fraction(2,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)])
sum_frac, sum_frac.numerator, sum_frac.denominator


### Turn 38
from fractions import Fraction
sum_frac = sum([Fraction(2, 3), Fraction(2, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)])
print((sum_frac, sum_frac.numerator, sum_frac.denominator))

### Turn 39
(224414944 + 57101475) % 99991


### Turn 39
print((224414944 + 57101475) % 99991)

### Turn 40
for c in c_labels:
    print("c=",c)
    for s in [11,7,17,10,14,23,13]:
        print(s, is_divisible_by_S(c, s))
    print()


### Turn 40
for c in c_labels:
    print('c=', c)
    for s in [11, 7, 17, 10, 14, 23, 13]:
        print(s, is_divisible_by_S(c, s))
    print()

