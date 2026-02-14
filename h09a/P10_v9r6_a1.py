# Auto-extracted from: P10_v9r6.md
# Attempt: 1

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
import math, itertools, sys, time, collections, random, functools, sympy as sp, math, fractions, itertools, collections, sys, math, sympy as sp

### Turn 0
import math, itertools, sys, time, collections, random, functools, sympy as sp, math, fractions, itertools, collections, sys, math, sympy as sp

### Turn 1
def f_bruteforce(n):
    best = None
    # distinct a<b<c positive integers sum n
    # naive triple loops O(n^2)
    for a in range(1, n-2):
        for b in range(a+1, n-a-1):
            c = n - a - b
            if c <= b:
                continue
            # compute lcm
            l = math.lcm(a,b,c)
            if best is None or l < best:
                best = l
    return best

def compute_f(N):
    vals={}
    for n in range(6,N+1):
        vals[n]=f_bruteforce(n)
    return vals

# test compute some
vals = compute_f(30)
for n in range(6,31):
    print(n, vals[n])


### Turn 1
def f_bruteforce(n):
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - a - 1):
            c = n - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if best is None or l < best:
                best = l
    return best

def compute_f(N):
    vals = {}
    for n in range(6, N + 1):
        vals[n] = f_bruteforce(n)
    return vals
vals = compute_f(30)
for n in range(6, 31):
    print(n, vals[n])

### Turn 2
def max_f_ratio(limit):
    mx = 0
    arg = None
    for n in range(6, limit+1):
        f = f_bruteforce(n)
        ratio = f / n
        if ratio > mx:
            mx = ratio
            arg = (n, f, ratio)
    return arg, mx

max_f_ratio(200)


### Turn 2
def max_f_ratio(limit):
    mx = 0
    arg = None
    for n in range(6, limit + 1):
        f = f_bruteforce(n)
        ratio = f / n
        if ratio > mx:
            mx = ratio
            arg = (n, f, ratio)
    return (arg, mx)
print(max_f_ratio(200))

### Turn 3
def compute_f_up_to(N):
    fvals = [0]*(N+1)
    for n in range(6,N+1):
        fvals[n]=f_bruteforce(n)
    return fvals

def max_f_ratio_up_to(N):
    max_ratio=0
    max_pair=None
    for n in range(6,N+1):
        f = f_bruteforce(n)
        ratio=f/n
        if ratio > max_ratio:
            max_ratio=ratio
            max_pair=(n,f,ratio)
    return max_pair

# Test up to 150 perhaps
max_f_ratio_up_to(150)


### Turn 3
def compute_f_up_to(N):
    fvals = [0] * (N + 1)
    for n in range(6, N + 1):
        fvals[n] = f_bruteforce(n)
    return fvals

def max_f_ratio_up_to(N):
    max_ratio = 0
    max_pair = None
    for n in range(6, N + 1):
        f = f_bruteforce(n)
        ratio = f / n
        if ratio > max_ratio:
            max_ratio = ratio
            max_pair = (n, f, ratio)
    return max_pair
print(max_f_ratio_up_to(150))

### Turn 4
def max_f_ratio_bruteforce(N):
    max_ratio=0
    max_pair=None
    for n in range(6,N+1):
        f=f_bruteforce(n)
        ratio=f/n
        if ratio>max_ratio:
            max_ratio=ratio
            max_pair=(n,f,ratio)
    return max_pair

max_f_ratio_bruteforce(300)


### Turn 4
def max_f_ratio_bruteforce(N):
    max_ratio = 0
    max_pair = None
    for n in range(6, N + 1):
        f = f_bruteforce(n)
        ratio = f / n
        if ratio > max_ratio:
            max_ratio = ratio
            max_pair = (n, f, ratio)
    return max_pair
print(max_f_ratio_bruteforce(300))

### Turn 5
def f_bruteforce_fast(n):
    # same as before
    best = None
    for a in range(1, n-2):
        for b in range(a+1, n-a-1):
            c=n-a-b
            if c<=b:
                continue
            l=math.lcm(a,b,c)
            if best is None or l<best:
                best=l
    return best

print("f(33) = ", f_bruteforce_fast(33))


### Turn 5
def f_bruteforce_fast(n):
    best = None
    for a in range(1, n - 2):
        for b in range(a + 1, n - a - 1):
            c = n - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if best is None or l < best:
                best = l
    return best
print('f(33) = ', f_bruteforce_fast(33))

### Turn 6
print(f_bruteforce_fast(44))


### Turn 6
print(f_bruteforce_fast(44))

### Turn 7
def min_ratio(Kmax):
    best=1
    best_pair=None
    for k in range(2,Kmax+1):
        for m in range(2,Kmax+1):
            if k==m: continue
            r = (k*m)/(k*m + k + m)
            if r < best:
                best = r
                best_pair=(k,m)
    return best, best_pair

print(min_ratio(50))


### Turn 7
def min_ratio(Kmax):
    best = 1
    best_pair = None
    for k in range(2, Kmax + 1):
        for m in range(2, Kmax + 1):
            if k == m:
                continue
            r = k * m / (k * m + k + m)
            if r < best:
                best = r
                best_pair = (k, m)
    return (best, best_pair)
print(min_ratio(50))

### Turn 8
def generate_fractions(limit):
    fracs={}
    for k in range(2, limit+1):
        for m in range(2, limit+1):
            if k==m: continue
            a = k*m
            b = k*m + k + m
            g = math.gcd(a,b)
            a_s = a//g
            b_s = b//g
            r = a_s / b_s
            # Keep minimal ratio for each denominator b_s
            if b_s not in fracs or r < fracs[b_s][0]:
                fracs[b_s] = (r, a_s, b_s, (k,m))
    return fracs

fracs = generate_fractions(50)
sorted_fracs = sorted(fracs.items(), key=lambda x: x[1][0])
sorted_fracs[:10]


### Turn 8
def generate_fractions(limit):
    fracs = {}
    for k in range(2, limit + 1):
        for m in range(2, limit + 1):
            if k == m:
                continue
            a = k * m
            b = k * m + k + m
            g = math.gcd(a, b)
            a_s = a // g
            b_s = b // g
            r = a_s / b_s
            if b_s not in fracs or r < fracs[b_s][0]:
                fracs[b_s] = (r, a_s, b_s, (k, m))
    return fracs
fracs = generate_fractions(50)
sorted_fracs = sorted(fracs.items(), key=lambda x: x[1][0])
print(head(sorted_fracs, 10))

### Turn 9
b_vals = set(fracs.keys())
sorted([b for b in b_vals if b%3==0])[:20]


### Turn 9
b_vals = set(fracs.keys())
print(head(sorted([b for b in b_vals if b % 3 == 0]), 20))

### Turn 10
sorted([(b,fracs[b]) for b in fracs if b%3==0])[:10]


### Turn 10
print(head(sorted([(b, fracs[b]) for b in fracs if b % 3 == 0]), 10))

### Turn 11
def get_fractions_upto_k(limit_k=100):
    fracs = []
    for k in range(2, limit_k+1):
        for m in range(2, limit_k+1):
            if k==m: continue
            a = k*m
            b = k*m + k + m
            g = math.gcd(a,b)
            a_s = a//g
            b_s = b//g
            fracs.append((a_s, b_s, k,m, a_s/b_s))
    # Remove duplicates for each b keep min ratio
    best = {}
    for a,b,k,m,r in fracs:
        if b not in best or a/b < best[b][0]:
            best[b] = (a/b, a, b, k, m)
    return best

best = get_fractions_upto_k(200)
# find powers of 3
powers_of_3 = [3**e for e in range(1,10)]
[(b, best[b]) for b in powers_of_3 if b in best]


### Turn 11
def get_fractions_upto_k(limit_k=100):
    fracs = []
    for k in range(2, limit_k + 1):
        for m in range(2, limit_k + 1):
            if k == m:
                continue
            a = k * m
            b = k * m + k + m
            g = math.gcd(a, b)
            a_s = a // g
            b_s = b // g
            fracs.append((a_s, b_s, k, m, a_s / b_s))
    best = {}
    for a, b, k, m, r in fracs:
        if b not in best or a / b < best[b][0]:
            best[b] = (a / b, a, b, k, m)
    return best
best = get_fractions_upto_k(200)
powers_of_3 = [3 ** e for e in range(1, 10)]
print([(b, best[b]) for b in powers_of_3 if b in best])

### Turn 12
cs = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 3 for c in cs]
mods


### Turn 12
cs = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 3 for c in cs]
print(mods)

### Turn 13
c_vals = [1848374,10162574,265710644,44636594]
for c in c_vals:
    s = c+1
    print(c, s, sp.factorint(s))


### Turn 13
c_vals = [1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    s = c + 1
    print(c, s, sp.factorint(s))

### Turn 14
def has_factor(n, p):
    return n % p == 0

c_factors = {}
for c in c_vals:
    c_factors[c] = sp.factorint(c+1)
c_factors


### Turn 14
def has_factor(n, p):
    return n % p == 0
c_factors = {}
for c in c_vals:
    c_factors[c] = sp.factorint(c + 1)
print(c_factors)

### Turn 15
sorted([b for b in best if b<=200 and b%5==0])[:20]


### Turn 15
print(head(sorted([b for b in best if b <= 200 and b % 5 == 0]), 20))

### Turn 16
best[25]


### Turn 16
print(best[25])

### Turn 17
primes = [7,11,17]
for c in c_vals:
    print("c:", c)
    for p in primes:
        print(p, (c+1) % p)


### Turn 17
primes = [7, 11, 17]
for c in c_vals:
    print('c:', c)
    for p in primes:
        print(p, (c + 1) % p)

### Turn 18
def f_min_lcm(n):
    # compute min lcm via pair method (optimal)
    # but for small n we brute force triple
    return f_bruteforce_fast(n)

def compute_g(c, M, factorial):
    n = M + c
    f_val = f_min_lcm(n)
    return math.floor(factorial * f_val / M) / factorial

M = 3**5 # 243
factorial = math.factorial(5) # 120
test_cs = [0, 4*M, 184, 1016, 26571] # using small values? Not same but just test with some
for c in test_cs:
    g_val = compute_g(c, M, factorial)
    print(c, g_val)


### Turn 18
def f_min_lcm(n):
    return f_bruteforce_fast(n)

def compute_g(c, M, factorial):
    n = M + c
    f_val = f_min_lcm(n)
    return math.floor(factorial * f_val / M) / factorial
M = 3 ** 5
factorial = math.factorial(5)
test_cs = [0, 4 * M, 184, 1016, 26571]
for c in test_cs:
    g_val = compute_g(c, M, factorial)
    print(c, g_val)

### Turn 19
def generate_fractions(limit=200):
    fracs = []
    for k in range(2, limit+1):
        for m in range(2, limit+1):
            if k==m:
                continue
            a = k*m
            b = k*m + k + m
            g = math.gcd(a,b)
            a_s = a//g
            b_s = b//g
            fracs.append((a_s, b_s, a_s/b_s))
    # keep minimal ratio for each denominator
    best_by_den = {}
    for a,b,r in fracs:
        if b not in best_by_den or r < best_by_den[b][0]:
            best_by_den[b] = (r, a, b)
    # return list of (a,b,r) sorted by ratio
    lst = [(a,b,r) for b,(r,a,b) in best_by_den.items()]
    lst.sort(key=lambda x: x[2])
    return lst

best_fracs = generate_fractions(200)
best_fracs[:10]


### Turn 19
def generate_fractions(limit=200):
    fracs = []
    for k in range(2, limit + 1):
        for m in range(2, limit + 1):
            if k == m:
                continue
            a = k * m
            b = k * m + k + m
            g = math.gcd(a, b)
            a_s = a // g
            b_s = b // g
            fracs.append((a_s, b_s, a_s / b_s))
    best_by_den = {}
    for a, b, r in fracs:
        if b not in best_by_den or r < best_by_den[b][0]:
            best_by_den[b] = (r, a, b)
    lst = [(a, b, r) for b, (r, a, b) in best_by_den.items()]
    lst.sort(key=lambda x: x[2])
    return lst
best_fracs = generate_fractions(200)
print(head(best_fracs, 10))

### Turn 20
[x for x in best_fracs if x[1]==3]


### Turn 20
print([x for x in best_fracs if x[1] == 3])

### Turn 21
def f_using_best(n):
    min_val = None
    min_frac = None
    for a,b,r in best_fracs:
        if n % b == 0:
            l = (a * (n // b))
            # Since ratio a/b times n: L = a/b * n = a * (n/b)
            if min_val is None or l < min_val:
                min_val = l
                min_frac = (a,b)
    return min_val, min_frac

def f_estimate(n):
    val, frac = f_using_best(n)
    return val

# Test small n to verify matches brute for n up to maybe 100
def test_small(N=100):
    for n in range(6,N+1):
        f_br = f_bruteforce_fast(n)
        f_est,frac = f_using_best(n)
        if f_br != f_est:
            print("Mismatch at n", n, f_br, f_est, frac)
            break
    else:
        print("All match up to", N)

test_small(200)


### Turn 21
def f_using_best(n):
    min_val = None
    min_frac = None
    for a, b, r in best_fracs:
        if n % b == 0:
            l = a * (n // b)
            if min_val is None or l < min_val:
                min_val = l
                min_frac = (a, b)
    return (min_val, min_frac)

def f_estimate(n):
    val, frac = f_using_best(n)
    return val

def test_small(N=100):
    for n in range(6, N + 1):
        f_br = f_bruteforce_fast(n)
        f_est, frac = f_using_best(n)
        if f_br != f_est:
            print('Mismatch at n', n, f_br, f_est, frac)
            break
    else:
        print('All match up to', N)
print(test_small(200))

### Turn 22
def compute_min_ratio_for_D(Dmax=5000):
    # compute all pairs (k,m) with D'= (km/g) + (k/g)+(m/g) where g=gcd(k,m)
    best = {}
    for k in range(2,200):
        for m in range(k+1,200):
            d = math.gcd(k,m)
            k1 = k//d
            m1 = m//d
            D_prime = d*k1*m1 + k1 + m1
            # Only consider D' <= Dmax
            if D_prime > Dmax:
                continue
            lcm = math.lcm(k,m)
            ratio = lcm / D_prime
            if D_prime not in best or ratio < best[D_prime][0]:
                best[D_prime] = (ratio, k, m, lcm)
    return best

best_by_D = compute_min_ratio_for_D(2000)
# find D' powers of 3 and minimal ratio
powers = [3**e for e in range(1,10)]
for D in powers:
    if D in best_by_D:
        print(D, best_by_D[D])


### Turn 22
def compute_min_ratio_for_D(Dmax=5000):
    best = {}
    for k in range(2, 200):
        for m in range(k + 1, 200):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            D_prime = d * k1 * m1 + k1 + m1
            if D_prime > Dmax:
                continue
            lcm = math.lcm(k, m)
            ratio = lcm / D_prime
            if D_prime not in best or ratio < best[D_prime][0]:
                best[D_prime] = (ratio, k, m, lcm)
    return best
best_by_D = compute_min_ratio_for_D(2000)
powers = [3 ** e for e in range(1, 10)]
for D in powers:
    if D in best_by_D:
        print(D, best_by_D[D])

### Turn 23
def find_pairs_D(target):
    results=[]
    for k in range(2,200):
        for m in range(k+1,200):
            d=math.gcd(k,m)
            k1=k//d
            m1=m//d
            Dp = d*k1*m1 + k1 + m1
            if Dp == target:
                results.append((k,m,math.lcm(k,m)))
    return results

print(find_pairs_D(5))


### Turn 23
def find_pairs_D(target):
    results = []
    for k in range(2, 200):
        for m in range(k + 1, 200):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            Dp = d * k1 * m1 + k1 + m1
            if Dp == target:
                results.append((k, m, math.lcm(k, m)))
    return results
print(find_pairs_D(5))

### Turn 24
print(find_pairs_D(15))


### Turn 24
print(find_pairs_D(15))

### Turn 25
def D_prime(k,m):
    d=math.gcd(k,m)
    k1=k//d
    m1=m//d
    return d*k1*m1 + k1 + m1

print(D_prime(2,5))


### Turn 25
def D_prime(k, m):
    d = math.gcd(k, m)
    k1 = k // d
    m1 = m // d
    return d * k1 * m1 + k1 + m1
print(D_prime(2, 5))

### Turn 26
pairs=[]
for k in range(2,200):
    for m in range(k+1,200):
        d=math.gcd(k,m)
        k1=k//d
        m1=m//d
        Dp = d*k1*m1 + k1 + m1
        l = math.lcm(k,m)
        if l * 5 == Dp * 3:
            pairs.append((k,m,l,Dp))
pairs[:10], len(pairs)


### Turn 26
pairs = []
for k in range(2, 200):
    for m in range(k + 1, 200):
        d = math.gcd(k, m)
        k1 = k // d
        m1 = m // d
        Dp = d * k1 * m1 + k1 + m1
        l = math.lcm(k, m)
        if l * 5 == Dp * 3:
            pairs.append((k, m, l, Dp))
print((head(pairs, 10), len(pairs)))

### Turn 27
print(D_prime(2,6), math.lcm(2,6))


### Turn 27
print(D_prime(2, 6), math.lcm(2, 6))

### Turn 28
def find_pairs_ratio(target_num, target_den, limit=200):
    results=[]
    for k in range(2,limit+1):
        for m in range(k+1,limit+1):
            d=math.gcd(k,m)
            k1=k//d
            m1=m//d
            Dp = d*k1*m1 + k1 + m1
            l = math.lcm(k,m)
            if l * target_den == Dp * target_num:
                # compute prime factorization of Dp
                factors = sp.factorint(Dp)
                results.append((k,m,l,Dp,factors))
    return results

pairs_3_5 = find_pairs_ratio(3,5,200)
len(pairs_3_5)


### Turn 28
def find_pairs_ratio(target_num, target_den, limit=200):
    results = []
    for k in range(2, limit + 1):
        for m in range(k + 1, limit + 1):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            Dp = d * k1 * m1 + k1 + m1
            l = math.lcm(k, m)
            if l * target_den == Dp * target_num:
                factors = sp.factorint(Dp)
                results.append((k, m, l, Dp, factors))
    return results
pairs_3_5 = find_pairs_ratio(3, 5, 200)
print(len(pairs_3_5))

### Turn 29
M_small = 3**6 # 729
n = 5*M_small
print("n=", n)
print("f_bruteforce:", f_bruteforce_fast(n))


### Turn 29
M_small = 3 ** 6
n = 5 * M_small
print('n=', n)
print('f_bruteforce:', f_bruteforce_fast(n))

### Turn 30
from fractions import Fraction
Fraction(2430,3645)


### Turn 30
from fractions import Fraction
print(Fraction(2430, 3645))

### Turn 31
fact = math.factorial(5)  # 120, divisible by 3? 120%3=0, yes
g_val = math.floor(fact * 2430 / M_small) / fact
g_val


### Turn 31
fact = math.factorial(5)
g_val = math.floor(fact * 2430 / M_small) / fact
print(g_val)

### Turn 32
def min_ratio_for_D(D, limit=200):
    best = None
    best_pair=None
    for k in range(2,limit+1):
        for m in range(k+1,limit+1):
            d=math.gcd(k,m)
            k1=k//d
            m1=m//d
            Dp = d*k1*m1 + k1 + m1
            if Dp == D:
                l = math.lcm(k,m)
                ratio = l / D
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (k,m,l)
    return best, best_pair

for D in [9,15,25,31,53,125]:
    r,p = min_ratio_for_D(D)
    print(D, r, p)


### Turn 32
def min_ratio_for_D(D, limit=200):
    best = None
    best_pair = None
    for k in range(2, limit + 1):
        for m in range(k + 1, limit + 1):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            Dp = d * k1 * m1 + k1 + m1
            if Dp == D:
                l = math.lcm(k, m)
                ratio = l / D
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (k, m, l)
    return (best, best_pair)
for D in [9, 15, 25, 31, 53, 125]:
    r, p = min_ratio_for_D(D)
    print(D, r, p)

### Turn 33
def compute_best_ratios(Dmax=2000, km_limit=200):
    best = {}
    for k in range(2, km_limit+1):
        for m in range(k+1, km_limit+1):
            d = math.gcd(k,m)
            k1 = k//d
            m1 = m//d
            Dp = d*k1*m1 + k1 + m1
            if Dp > Dmax:
                continue
            l = math.lcm(k,m)
            ratio = l / Dp
            if Dp not in best or ratio < best[Dp][0]:
                best[Dp] = (ratio, k, m, l)
    return best

best_by_D = compute_best_ratios(5000,200)
len(best_by_D)


### Turn 33
def compute_best_ratios(Dmax=2000, km_limit=200):
    best = {}
    for k in range(2, km_limit + 1):
        for m in range(k + 1, km_limit + 1):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            Dp = d * k1 * m1 + k1 + m1
            if Dp > Dmax:
                continue
            l = math.lcm(k, m)
            ratio = l / Dp
            if Dp not in best or ratio < best[Dp][0]:
                best[Dp] = (ratio, k, m, l)
    return best
best_by_D = compute_best_ratios(5000, 200)
print(len(best_by_D))

### Turn 34
def min_ratio_for_n(c):
    n_mod = c+1
    # get divisors of n_mod <=5000
    divs = sp.divisors(n_mod)
    min_ratio = None
    best_info = None
    for d in divs:
        if d % 2 == 0:  # D' must be odd because n odd, D' must be odd divisor
            continue
        if d in best_by_D:
            ratio, k, m, l = best_by_D[d]
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                best_info = (d, ratio, k, m, l)
    return min_ratio, best_info

for c in c_vals:
    r, info = min_ratio_for_n(c)
    print(c, r, info)


### Turn 34
def min_ratio_for_n(c):
    n_mod = c + 1
    divs = sp.divisors(n_mod)
    min_ratio = None
    best_info = None
    for d in divs:
        if d % 2 == 0:
            continue
        if d in best_by_D:
            ratio, k, m, l = best_by_D[d]
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                best_info = (d, ratio, k, m, l)
    return (min_ratio, best_info)
for c in c_vals:
    r, info = min_ratio_for_n(c)
    print(c, r, info)

### Turn 35
def min_ratio_for_n_product_of_primes(prime_factors):
    # prime_factors dict of prime: exponent, e.g., {3:exp,5:exp}
    # generate divisors up to limit
    limit = 5000
    # generate all odd divisors <= limit
    divisors = [1]
    for p,exp in prime_factors.items():
        new_divs = []
        for d in divisors:
            for e in range(1,exp+1):
                new_divs.append(d * (p**e))
        divisors.extend(new_divs)
    # unique
    divisors = set(divisors)
    min_ratio = None
    best=None
    for D in divisors:
        if D%2==0:
            continue
        if D in best_by_D:
            ratio,k,m,l = best_by_D[D][0],best_by_D[D][1],best_by_D[D][2],best_by_D[D][3]
            if min_ratio is None or ratio < min_ratio:
                min_ratio=ratio
                best=(D,ratio,k,m,l)
    return min_ratio,best

# M includes prime 3 with huge exponent, treat as at least exponent 10
prime_factors_M = {3:10}
# n=5M includes prime factor 5 exponent 1
prime_factors_5M = {3:10,5:1}
min_ratio_M = min_ratio_for_n_product_of_primes(prime_factors_M)
min_ratio_5M = min_ratio_for_n_product_of_primes(prime_factors_5M)
min_ratio_M, min_ratio_5M


### Turn 35
def min_ratio_for_n_product_of_primes(prime_factors):
    limit = 5000
    divisors = [1]
    for p, exp in prime_factors.items():
        new_divs = []
        for d in divisors:
            for e in range(1, exp + 1):
                new_divs.append(d * p ** e)
        divisors.extend(new_divs)
    divisors = set(divisors)
    min_ratio = None
    best = None
    for D in divisors:
        if D % 2 == 0:
            continue
        if D in best_by_D:
            ratio, k, m, l = (best_by_D[D][0], best_by_D[D][1], best_by_D[D][2], best_by_D[D][3])
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                best = (D, ratio, k, m, l)
    return (min_ratio, best)
prime_factors_M = {3: 10}
prime_factors_5M = {3: 10, 5: 1}
min_ratio_M = min_ratio_for_n_product_of_primes(prime_factors_M)
min_ratio_5M = min_ratio_for_n_product_of_primes(prime_factors_5M)
print((min_ratio_M, min_ratio_5M))

### Turn 36
M_small=3**6
n=5*M_small
f_val=2430
fact=math.factorial(5)
g_val=math.floor(fact*f_val/M_small)/fact
g_val


### Turn 36
M_small = 3 ** 6
n = 5 * M_small
f_val = 2430
fact = math.factorial(5)
g_val = math.floor(fact * f_val / M_small) / fact
print(g_val)

### Turn 37
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S


### Turn 37
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print(S)

### Turn 38
import math
p = S.numerator
q = S.denominator
math.gcd(p,q)


### Turn 38
import math
p = S.numerator
q = S.denominator
print(math.gcd(p, q))

### Turn 39
mod = 99991
(p+q) % mod


### Turn 39
mod = 99991
print((p + q) % mod)

### Turn 40
def f_estimate_via_D(n, best_by_D):
    # find divisors d of n
    divs = sp.divisors(n)
    min_val = None
    min_info = None
    for D in divs:
        if D % 2 ==0: continue  # D must be odd? Not necessarily if n odd? Actually D' always odd? For n odd, D' must be odd as D' divides n and is odd (since D' = d*k1*m1 + k1 + m1, always odd? Let's ignore parity.
        if D in best_by_D:
            l = best_by_D[D][3]  # lcm
            val = (l * n) // D
            if min_val is None or val < min_val:
                min_val = val
                min_info = (D, l, val)
    return min_val, min_info

def f_brute(n):
    return f_bruteforce_fast(n)

def test_random(N=500):
    for n in range(6,N+1):
        val_est, info = f_estimate_via_D(n, best_by_D)
        val_brute = f_brute(n)
        if val_est != val_brute:
            print("Mismatch at n", n, val_est, val_brute, info)
            return False
    print("All matches up to", N)
    return True

test_random(200)


### Turn 40
def f_estimate_via_D(n, best_by_D):
    divs = sp.divisors(n)
    min_val = None
    min_info = None
    for D in divs:
        if D % 2 == 0:
            continue
        if D in best_by_D:
            l = best_by_D[D][3]
            val = l * n // D
            if min_val is None or val < min_val:
                min_val = val
                min_info = (D, l, val)
    return (min_val, min_info)

def f_brute(n):
    return f_bruteforce_fast(n)

def test_random(N=500):
    for n in range(6, N + 1):
        val_est, info = f_estimate_via_D(n, best_by_D)
        val_brute = f_brute(n)
        if val_est != val_brute:
            print('Mismatch at n', n, val_est, val_brute, info)
            return False
    print('All matches up to', N)
    return True
print(test_random(200))

### Turn 41
def test_odds(N=200):
    for n in range(7,N+1,2):
        val_est,_ = f_estimate_via_D(n,best_by_D)
        val_brute = f_brute(n)
        if val_est != val_brute:
            print("Mismatch n",n, val_est, val_brute)
            return False
    print("All odd matches up to", N)
    return True

test_odds(199)


### Turn 41
def test_odds(N=200):
    for n in range(7, N + 1, 2):
        val_est, _ = f_estimate_via_D(n, best_by_D)
        val_brute = f_brute(n)
        if val_est != val_brute:
            print('Mismatch n', n, val_est, val_brute)
            return False
    print('All odd matches up to', N)
    return True
print(test_odds(199))

### Turn 42
for c in c_vals:
    n = c+1
    print(c, n % 11)


### Turn 42
for c in c_vals:
    n = c + 1
    print(c, n % 11)

### Turn 43
for c in c_vals:
    print(c, (c+1)%7)


### Turn 43
for c in c_vals:
    print(c, (c + 1) % 7)

### Turn 44
for c in c_vals:
    print(c, (c+1)%17)


### Turn 44
for c in c_vals:
    print(c, (c + 1) % 17)

### Turn 45
for c in c_vals:
    print(c, (c+1)%31)


### Turn 45
for c in c_vals:
    print(c, (c + 1) % 31)

### Turn 46
def best_ratio_for_prime(p):
    # find best ratio for D'=p among pairs
    min_ratio = None
    pair = None
    for k in range(2,500):
        for m in range(k+1,500):
            d = math.gcd(k,m)
            k1 = k//d
            m1 = m//d
            Dp = d*k1*m1 + k1 + m1
            if Dp == p:
                l = math.lcm(k,m)
                ratio = l / p
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    pair = (k,m,l)
    return min_ratio, pair

primes = list(sp.primerange(3,200))
best_primes = []
for p in primes:
    r,pair = best_ratio_for_prime(p)
    if r is not None:
        best_primes.append((p,r,pair))
sorted(best_primes, key=lambda x: x[1])[:10]


### Turn 46
def best_ratio_for_prime(p):
    min_ratio = None
    pair = None
    for k in range(2, 500):
        for m in range(k + 1, 500):
            d = math.gcd(k, m)
            k1 = k // d
            m1 = m // d
            Dp = d * k1 * m1 + k1 + m1
            if Dp == p:
                l = math.lcm(k, m)
                ratio = l / p
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    pair = (k, m, l)
    return (min_ratio, pair)
primes = list(sp.primerange(3, 200))
best_primes = []
for p in primes:
    r, pair = best_ratio_for_prime(p)
    if r is not None:
        best_primes.append((p, r, pair))
print(head(sorted(best_primes, key=lambda x: x[1]), 10))

### Turn 47
for c in c_vals:
    print(c, (c+1)%23)


### Turn 47
for c in c_vals:
    print(c, (c + 1) % 23)

### Turn 48
for c in c_vals:
    print(c, (c+1)%35)


### Turn 48
for c in c_vals:
    print(c, (c + 1) % 35)

### Turn 49
r, info = min_ratio_for_D(45)
r, info


### Turn 49
r, info = min_ratio_for_D(45)
print((r, info))

### Turn 50
r,info = min_ratio_for_D(65)
r, info


### Turn 50
r, info = min_ratio_for_D(65)
print((r, info))

### Turn 51
for c in c_vals:
    print(c, (c+1)%41)


### Turn 51
for c in c_vals:
    print(c, (c + 1) % 41)

### Turn 52
r,info=min_ratio_for_D(95)
r,info


### Turn 52
r, info = min_ratio_for_D(95)
print((r, info))

### Turn 53
r,info=min_ratio_for_D(91)
r,info


### Turn 53
r, info = min_ratio_for_D(91)
print((r, info))

### Turn 54
for c in c_vals:
    print(c, (c+1)%91)


### Turn 54
for c in c_vals:
    print(c, (c + 1) % 91)

### Turn 55
def best_ratio_via_all_divisors(c):
    n = c+1
    divs = sp.divisors(n)
    best_ratio = None
    best_D = None
    best_l = None
    for D in divs:
        if D % 2 == 0:
            continue # D must be odd because n is odd? but D can be even? D'=D? Actually D' can be even? For n odd, D' odd.
        # Need to find minimal ratio for this D: we must compute minimal l for this D.
        # We'll compute via brute over pairs up to some limit.
        # Instead use our best_by_D if exists; else compute minimal for this D via search.
        if D in best_by_D:
            ratio, k,m,l = best_by_D[D][0], best_by_D[D][1], best_by_D[D][2], best_by_D[D][3]
        else:
            # brute search for minimal l for given D
            min_ratio = None
            min_l = None
            min_pair = None
            for k in range(2,200):
                for m in range(k+1,200):
                    d = math.gcd(k,m)
                    k1 = k//d
                    m1 = m//d
                    Dp = d*k1*m1 + k1 + m1
                    if Dp == D:
                        lcm = math.lcm(k,m)
                        ratio = lcm/D
                        if min_ratio is None or ratio < min_ratio:
                            min_ratio = ratio
                            min_l = lcm
                            min_pair = (k,m)
            if min_ratio is None:
                continue
            ratio = min_ratio
            k,m = min_pair
            l = min_l
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_D = D
            best_l = l
    return best_ratio, best_D, best_l

for c in c_vals:
    br, D, l = best_ratio_via_all_divisors(c)
    print("c",c,"best ratio", br, "D", D, "l",l)


### Turn 55
def best_ratio_via_all_divisors(c):
    n = c + 1
    divs = sp.divisors(n)
    best_ratio = None
    best_D = None
    best_l = None
    for D in divs:
        if D % 2 == 0:
            continue
        if D in best_by_D:
            ratio, k, m, l = (best_by_D[D][0], best_by_D[D][1], best_by_D[D][2], best_by_D[D][3])
        else:
            min_ratio = None
            min_l = None
            min_pair = None
            for k in range(2, 200):
                for m in range(k + 1, 200):
                    d = math.gcd(k, m)
                    k1 = k // d
                    m1 = m // d
                    Dp = d * k1 * m1 + k1 + m1
                    if Dp == D:
                        lcm = math.lcm(k, m)
                        ratio = lcm / D
                        if min_ratio is None or ratio < min_ratio:
                            min_ratio = ratio
                            min_l = lcm
                            min_pair = (k, m)
            if min_ratio is None:
                continue
            ratio = min_ratio
            k, m = min_pair
            l = min_l
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_D = D
            best_l = l
    return (best_ratio, best_D, best_l)
for c in c_vals:
    br, D, l = best_ratio_via_all_divisors(c)
    print('c', c, 'best ratio', br, 'D', D, 'l', l)

### Turn 56
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S, S.numerator, S.denominator


### Turn 56
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print((S, S.numerator, S.denominator))

### Turn 57
(p+q) % 99991


### Turn 57
print((p + q) % 99991)

### Turn 58
best_by_D.get(3)


### Turn 58
print(best_by_D.get(3))

### Turn 59
print(best_by_D[3])


### Turn 59
print(best_by_D[3])

### Turn 60
sorted([d for d in best_by_D.keys() if d<100])[:20]


### Turn 60
print(head(sorted([d for d in best_by_D.keys() if d < 100]), 20))

### Turn 61
from fractions import Fraction
terms = [Fraction(2,3), Fraction(10,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)]
S = sum(terms, Fraction(0,1))
S, S.numerator, S.denominator


### Turn 61
from fractions import Fraction
terms = [Fraction(2, 3), Fraction(10, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)]
S = sum(terms, Fraction(0, 1))
print((S, S.numerator, S.denominator))

### Turn 62
(p+q) % 99991


### Turn 62
print((p + q) % 99991)

