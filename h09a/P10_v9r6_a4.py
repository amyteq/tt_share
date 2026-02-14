# Auto-extracted from: P10_v9r6.md
# Attempt: 4

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

def f_brutal(n):
    min_lcm = None
    best = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c<=b or c<=0: continue
            # distinct guaranteed
            l = math.lcm(a,b,c)
            if min_lcm is None or l < min_lcm:
                min_lcm = l
                best = (a,b,c)
    return min_lcm, best

def compute_f_range(limit):
    data = {}
    for n in range(6, limit+1):
        data[n] = f_brutal(n)[0]
    return data

data = compute_f_range(100)
data[:10]


### Turn 0
import math, itertools

def f_brutal(n):
    min_lcm = None
    best = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b or c <= 0:
                continue
            l = math.lcm(a, b, c)
            if min_lcm is None or l < min_lcm:
                min_lcm = l
                best = (a, b, c)
    return (min_lcm, best)

def compute_f_range(limit):
    data = {}
    for n in range(6, limit + 1):
        data[n] = f_brutal(n)[0]
    return data
data = compute_f_range(100)
print(head(data, 10))

### Turn 1
for n in range(6, 31):
    print(n, data[n])


### Turn 1
for n in range(6, 31):
    print(n, data[n])

### Turn 2
for n in range(6,31):
    l,b = f_brutal(n)
    print(n, l, b)


### Turn 2
for n in range(6, 31):
    l, b = f_brutal(n)
    print(n, l, b)

### Turn 3
def compute_f_and_best(limit):
    fvals={}
    besttrip={}
    for n in range(6, limit+1):
        l,b = f_brutal(n)
        fvals[n]=l
        besttrip[n]=b
    return fvals, besttrip

fvals,besttrip=compute_f_and_best(200)
for n in range(6,51):
    print(n, fvals[n], besttrip[n])


### Turn 3
def compute_f_and_best(limit):
    fvals = {}
    besttrip = {}
    for n in range(6, limit + 1):
        l, b = f_brutal(n)
        fvals[n] = l
        besttrip[n] = b
    return (fvals, besttrip)
fvals, besttrip = compute_f_and_best(200)
for n in range(6, 51):
    print(n, fvals[n], besttrip[n])

### Turn 4
c_vals={}
for n in range(6,101):
    c_vals[n]=max(besttrip[n])
for n in range(6,31):
    print(n, c_vals[n])


### Turn 4
c_vals = {}
for n in range(6, 101):
    c_vals[n] = max(besttrip[n])
for n in range(6, 31):
    print(n, c_vals[n])

### Turn 5
1848374 % 3


### Turn 5
print(1848374 % 3)

### Turn 6
10162574 % 3


### Turn 6
print(10162574 % 3)

### Turn 7
265710644 % 3


### Turn 7
print(265710644 % 3)

### Turn 8
44636594 % 3


### Turn 8
print(44636594 % 3)

### Turn 9
1848374 % 4


### Turn 9
print(1848374 % 4)

### Turn 10
10162574 % 4


### Turn 10
print(10162574 % 4)

### Turn 11
265710644 % 4


### Turn 11
print(265710644 % 4)

### Turn 12
44636594 % 4


### Turn 12
print(44636594 % 4)

### Turn 13
1848374 % 5


### Turn 13
print(1848374 % 5)

### Turn 14
10162574 % 5


### Turn 14
print(10162574 % 5)

### Turn 15
265710644 % 5


### Turn 15
print(265710644 % 5)

### Turn 16
44636594 % 5


### Turn 16
print(44636594 % 5)

### Turn 17
def mods(c,mod):
    return c%mod
cs=[0,1848374,10162574,265710644,44636594]
for mod in [7,8,9,10,11,12,13,14,15]:
    print("mod",mod,[c%mod for c in cs])


### Turn 17
def mods(c, mod):
    return c % mod
cs = [0, 1848374, 10162574, 265710644, 44636594]
for mod in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
    print('mod', mod, [c % mod for c in cs])

### Turn 18
cs=[1848374,10162574,265710644,44636594]
[c%16 for c in cs]


### Turn 18
cs = [1848374, 10162574, 265710644, 44636594]
print([c % 16 for c in cs])

### Turn 19
[c%17 for c in cs]


### Turn 19
print([c % 17 for c in cs])

### Turn 20
import math

K = math.factorial(2025)  # too huge, can't compute! 


### Turn 20
import math
print(K)

### Turn 21
import math, sys, time, os, gc, sys, resource, numpy as np
K = math.factorial(2025)
len_str = K.bit_length()
len_str


### Turn 21
import math, sys, time, os, gc, sys, resource, numpy as np
K = math.factorial(2025)
len_str = K.bit_length()
print(len_str)

### Turn 22
def f_via_formula(n):
    if n<=5: 
        return None
    target = n-1
    # find smallest divisor >=3
    m = None
    # find divisors up to sqrt maybe but we just iterate small numbers
    # but need smallest divisor >=3; start from 3 upward
    for d in range(3, int(math.isqrt(target))+1):
        if target % d == 0:
            m = d
            break
    if m is None:
        # maybe target itself prime or 2 times prime? need check if target is even: then m could be target/2 (since we need divisor >=3)
        # Actually if target is even and not divisible by 3 or other small, smallest divisor >=3 might be target/2 if target/2 >=3.
        # That is when target is even and its only divisors are 1,2,target.
        # Then smallest divisor >=3 would be target if target >=3 (which is >2).
        # But that yields m=target, leading d = 1.
        m = target
    # Now compute f
    return (target)*(m-1)//m

def compare(limit):
    ok=True
    mism=[]
    for n in range(6,limit+1):
        f_real,_ = f_brutal(n)
        f_form = f_via_formula(n)
        if f_real != f_form:
            ok=False
            mism.append((n,f_real,f_form))
    return ok,mism

ok,mism=compare(200)
ok, mism[:10]


### Turn 22
def f_via_formula(n):
    if n <= 5:
        return None
    target = n - 1
    m = None
    for d in range(3, int(math.isqrt(target)) + 1):
        if target % d == 0:
            m = d
            break
    if m is None:
        m = target
    return target * (m - 1) // m

def compare(limit):
    ok = True
    mism = []
    for n in range(6, limit + 1):
        f_real, _ = f_brutal(n)
        f_form = f_via_formula(n)
        if f_real != f_form:
            ok = False
            mism.append((n, f_real, f_form))
    return (ok, mism)
ok, mism = compare(200)
print((ok, head(mism, 10)))

### Turn 23
def find_m_smallest_divisor_gte3(num):
    # returns smallest divisor >=3 of num, else returns num if none (i.e., prime)
    for d in range(3, int(math.isqrt(num))+1):
        if num % d == 0:
            return d
    return num  # prime or 2 * prime? For even numbers may have 2 only divisor, then returns num

def f_formula(n):
    target=n-1
    m=find_m_smallest_divisor_gte3(target)
    # Ensure m>=3; if target is even but only divisor 2, then m=target (odd? maybe prime)
    # Check condition d = target/m must be >=2 (since d must be proper divisor >1). If m==target, then d=1 => invalid.
    if m==target:
        # fallback: need other method.
        return None
    # compute N = target - target/m = target*(m-1)/m
    return target * (m-1) // m

def brute_f(n):
    return f_brutal(n)[0]

def test_range(lo,hi):
    mism=[]
    for n in range(lo, hi+1):
        real = brute_f(n)
        pred = f_formula(n)
        if pred is None or real!=pred:
            mism.append((n,real,pred))
    return mism

mism = test_range(100,150)
len(mism), mism[:10]


### Turn 23
def find_m_smallest_divisor_gte3(num):
    for d in range(3, int(math.isqrt(num)) + 1):
        if num % d == 0:
            return d
    return num

def f_formula(n):
    target = n - 1
    m = find_m_smallest_divisor_gte3(target)
    if m == target:
        return None
    return target * (m - 1) // m

def brute_f(n):
    return f_brutal(n)[0]

def test_range(lo, hi):
    mism = []
    for n in range(lo, hi + 1):
        real = brute_f(n)
        pred = f_formula(n)
        if pred is None or real != pred:
            mism.append((n, real, pred))
    return mism
mism = test_range(100, 150)
print((len(mism), head(mism, 10)))

### Turn 24
def f_one_method(n):
    target=n-1
    # find smallest divisor >=3
    for d in range(3, int(math.isqrt(target))+1):
        if target % d ==0:
            m = d
            break
    else:
        # target prime => not applicable
        return None
    # compute N = target - target//m
    return target - target//m

def compare_random(limit):
    mism=[]
    for n in range(6, limit+1):
        real=brute_f(n)
        approx=f_one_method(n)
        if approx is None:
            continue
        if approx < real:
            mism.append((n,real,approx))
    return mism[:10]

compare_random(200)


### Turn 24
def f_one_method(n):
    target = n - 1
    for d in range(3, int(math.isqrt(target)) + 1):
        if target % d == 0:
            m = d
            break
    else:
        return None
    return target - target // m

def compare_random(limit):
    mism = []
    for n in range(6, limit + 1):
        real = brute_f(n)
        approx = f_one_method(n)
        if approx is None:
            continue
        if approx < real:
            mism.append((n, real, approx))
    return head(mism, 10)
print(compare_random(200))

### Turn 25
def get_m_smallest_divisor_ge3(num):
    for d in range(3, int(math.isqrt(num))+1):
        if num % d == 0:
            return d
    return None # None for prime or 2 only

def test_m3or4(limit):
    better=[]
    for n in range(6, limit+1):
        target=n-1
        m=get_m_smallest_divisor_ge3(target)
        if m in [3,4]:
            N_candidate = target - target//m
            real=brute_f(n)
            if real < N_candidate:
                better.append((n, m, real, N_candidate))
    return better

better = test_m3or4(500)
len(better)


### Turn 25
def get_m_smallest_divisor_ge3(num):
    for d in range(3, int(math.isqrt(num)) + 1):
        if num % d == 0:
            return d
    return None

def test_m3or4(limit):
    better = []
    for n in range(6, limit + 1):
        target = n - 1
        m = get_m_smallest_divisor_ge3(target)
        if m in [3, 4]:
            N_candidate = target - target // m
            real = brute_f(n)
            if real < N_candidate:
                better.append((n, m, real, N_candidate))
    return better
better = test_m3or4(500)
print(len(better))

### Turn 26
better[:20]


### Turn 26
print(head(better, 20))

### Turn 27
vals = [(n, f_brutal(n)[0]/n) for n in range(6, 201)]
min(vals, key=lambda x: x[1]), max(vals, key=lambda x: x[1])


### Turn 27
vals = [(n, f_brutal(n)[0] / n) for n in range(6, 201)]
print((min(vals, key=lambda x: x[1]), max(vals, key=lambda x: x[1])))

### Turn 28
def compute_f_upto(N):
    return [f_brutal(n)[0] for n in range(6,N+1)]

fvals_500 = compute_f_upto(500)
ratios = [fvals_500[i-6]/i for i in range(6,501)]
min_ratio = min(ratios)
max_ratio = max(ratios)
(min_ratio, max_ratio)


### Turn 28
def compute_f_upto(N):
    return [f_brutal(n)[0] for n in range(6, N + 1)]
fvals_500 = compute_f_upto(500)
ratios = [fvals_500[i - 6] / i for i in range(6, 501)]
min_ratio = min(ratios)
max_ratio = max(ratios)
print((min_ratio, max_ratio))

### Turn 29
def n_with_m3(limit):
    diffs=[]
    for n in range(6, limit+1):
        target=n-1
        if target % 3 == 0:
            N1 = target - target//3  # 2/3 target
            real = brute_f(n)
            diffs.append((n, real, N1, real/N1))
    return diffs

diffs=n_with_m3(200)
# count where real < N1
cnt = sum(1 for _,real,N1,_ in diffs if real < N1)
cnt, len(diffs)


### Turn 29
def n_with_m3(limit):
    diffs = []
    for n in range(6, limit + 1):
        target = n - 1
        if target % 3 == 0:
            N1 = target - target // 3
            real = brute_f(n)
            diffs.append((n, real, N1, real / N1))
    return diffs
diffs = n_with_m3(200)
cnt = sum((1 for _, real, N1, _ in diffs if real < N1))
print((cnt, len(diffs)))

### Turn 30
def proper_divisors(n):
    # return sorted list of proper divisors (excluding n)
    divs=set()
    for i in range(1,int(math.isqrt(n))+1):
        if n%i==0:
            divs.add(i)
            divs.add(n//i)
    divs.discard(n)
    return sorted(divs)

def sums_of_two_distinct(divs):
    s=set()
    L=len(divs)
    for i in range(L):
        for j in range(i+1,L):
            s.add(divs[i]+divs[j])
    return s

def check_interval(N):
    divs = proper_divisors(N)
    sums = sums_of_two_distinct(divs)
    tmax = max(sums) if sums else 0
    # compute minimal possible sum (since need distinct and at least 1+ smallest proper divisor)
    # But just return range
    return tmax, sums

def test_range(N):
    tmax, sums = check_interval(N)
    # Determine if all integers from min_sum to tmax are present
    min_sum = min(sums) if sums else None
    missing=[x for x in range(min_sum, tmax+1) if x not in sums]
    return min_sum, tmax, missing[:10]


### Turn 30
def proper_divisors(n):
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(n)
    return sorted(divs)

def sums_of_two_distinct(divs):
    s = set()
    L = len(divs)
    for i in range(L):
        for j in range(i + 1, L):
            s.add(divs[i] + divs[j])
    return s

def check_interval(N):
    divs = proper_divisors(N)
    sums = sums_of_two_distinct(divs)
    tmax = max(sums) if sums else 0
    return (tmax, sums)

def test_range(N):
    tmax, sums = check_interval(N)
    min_sum = min(sums) if sums else None
    missing = [x for x in range(min_sum, tmax + 1) if x not in sums]
    return (min_sum, tmax, head(missing, 10))

### Turn 31
for N in [6,12,18,30,60,84,120]:
    mins, tmax, missing = test_range(N)
    print(N, mins, tmax, len(missing))


### Turn 31
for N in [6, 12, 18, 30, 60, 84, 120]:
    mins, tmax, missing = test_range(N)
    print(N, mins, tmax, len(missing))

### Turn 32
def f_estimate_via_6(n):
    # compute lower bound N0 = ceil(6n/11)
    N0 = (6*n + 10)//11  # ceil
    # round up to next multiple of 6
    if N0 % 6 != 0:
        N0 = N0 + (6 - N0%6)
    # Search upward for N meeting condition.
    # We'll limit search up to maybe N0+200 (should be enough)
    for N in range(N0, N0+1000):
        if N % 6 !=0:
            continue
        # compute t = n - N
        t = n - N
        if t <= 0:
            continue
        # compute sums of two distinct proper divisors of N
        divs = proper_divisors(N)
        # Use set for quick check
        found=False
        # Optimize: check for each divisor a, see if t-a is divisor distinct and not equal a.
        div_set=set(divs)
        for a in divs:
            b = t - a
            if b!=a and b in div_set:
                # ensure b != N
                if a!=N and b!=N:
                    found=True
                    break
        if found:
            return N
    return None

# Test for n up to 200 compare to real f
mismatches=[]
for n in range(6,201):
    est = f_estimate_via_6(n)
    real = f_brutal(n)[0]
    if est != real:
        mismatches.append((n, real, est))
len(mismatches), mismatches[:20]


### Turn 32
def f_estimate_via_6(n):
    N0 = (6 * n + 10) // 11
    if N0 % 6 != 0:
        N0 = N0 + (6 - N0 % 6)
    for N in range(N0, N0 + 1000):
        if N % 6 != 0:
            continue
        t = n - N
        if t <= 0:
            continue
        divs = proper_divisors(N)
        found = False
        div_set = set(divs)
        for a in divs:
            b = t - a
            if b != a and b in div_set:
                if a != N and b != N:
                    found = True
                    break
        if found:
            return N
    return None
mismatches = []
for n in range(6, 201):
    est = f_estimate_via_6(n)
    real = f_brutal(n)[0]
    if est != real:
        mismatches.append((n, real, est))
print((len(mismatches), head(mismatches, 20)))

### Turn 33
def f_brutal_by_N(n):
    # search over N from ceil(n/3) up to n-3
    min_N = None
    for N in range(math.ceil(n/3), n):
        # compute set of sums of two distinct proper divisors of N
        divs = proper_divisors(N)
        # need two distinct proper divisors a,b such that a+b = n - N
        target = n - N
        found=False
        div_set=set(divs)
        for a in divs:
            b = target - a
            if b!=a and b in div_set:
                found=True
                break
        if found:
            min_N=N
            break
    return min_N

def heuristic_N(n):
    N0 = (6*n + 10)//11  # ceil(6n/11)
    if N0%6 !=0:
        N0 += (6 - N0%6)
    # check if this N works; if not, search upward multiples of 6
    N=N0
    while True:
        target = n - N
        if target>0:
            divs=proper_divisors(N)
            div_set=set(divs)
            for a in divs:
                b = target - a
                if b!=a and b in div_set:
                    return N
        N+=6  # next multiple of 6

# test for n up to 200 maybe
mismatch2=[]
for n in range(6,201):
    real=f_brutal_by_N(n)
    heur=heuristic_N(n)
    if real!=heur:
        mismatch2.append((n,real,heur))
len(mismatch2), mismatch2[:20]


### Turn 33
def f_brutal_by_N(n):
    min_N = None
    for N in range(math.ceil(n / 3), n):
        divs = proper_divisors(N)
        target = n - N
        found = False
        div_set = set(divs)
        for a in divs:
            b = target - a
            if b != a and b in div_set:
                found = True
                break
        if found:
            min_N = N
            break
    return min_N

def heuristic_N(n):
    N0 = (6 * n + 10) // 11
    if N0 % 6 != 0:
        N0 += 6 - N0 % 6
    N = N0
    while True:
        target = n - N
        if target > 0:
            divs = proper_divisors(N)
            div_set = set(divs)
            for a in divs:
                b = target - a
                if b != a and b in div_set:
                    return N
        N += 6
mismatch2 = []
for n in range(6, 201):
    real = f_brutal_by_N(n)
    heur = heuristic_N(n)
    if real != heur:
        mismatch2.append((n, real, heur))
print((len(mismatch2), head(mismatch2, 20)))

### Turn 34
def brute_f_until(N_limit, n):
    # Search N from lower bound ceil(6n/11) up to N_limit (maybe n)
    lower = (6*n + 10)//11
    if lower % 6 != 0:
        lower += (6 - lower%6)
    for N in range(lower, n):
        # check if N works
        t = n - N
        if t <= 0: continue
        divs = proper_divisors(N)
        div_set=set(divs)
        found=False
        for a in divs:
            b = t - a
            if b != a and b in div_set:
                found=True
                break
        if found:
            return N
    return None

def test_range_brute(start,end):
    mism=[]
    for n in range(start,end+1):
        real=f_brutal_by_N(n)  # earlier exhaustive slower but ok up to maybe 200
        heur=heuristic_N(n)
        if real!=heur:
            mism.append((n,real,heur))
    return mism

# test for n from 100 to 200
mismatches = test_range_brute(100,150)
len(mismatches), mismatches[:10]


### Turn 34
def brute_f_until(N_limit, n):
    lower = (6 * n + 10) // 11
    if lower % 6 != 0:
        lower += 6 - lower % 6
    for N in range(lower, n):
        t = n - N
        if t <= 0:
            continue
        divs = proper_divisors(N)
        div_set = set(divs)
        found = False
        for a in divs:
            b = t - a
            if b != a and b in div_set:
                found = True
                break
        if found:
            return N
    return None

def test_range_brute(start, end):
    mism = []
    for n in range(start, end + 1):
        real = f_brutal_by_N(n)
        heur = heuristic_N(n)
        if real != heur:
            mism.append((n, real, heur))
    return mism
mismatches = test_range_brute(100, 150)
print((len(mismatches), head(mismatches, 10)))

### Turn 35
K = math.factorial(2025)
M_mod_11 = pow(3, K, 11)
M_mod_11


### Turn 35
K = math.factorial(2025)
M_mod_11 = pow(3, K, 11)
print(M_mod_11)

### Turn 36
c_vals=[0, 4* (None), 1848374, 10162574, 265710644, 44636594]
# compute M_mod_11 already
def n_mod_11(c):
    # c could be huge (like 4M). We'll handle for c=4M separately.
    pass

### Turn 36
c_vals = [0, 4 * None, 1848374, 10162574, 265710644, 44636594]

def n_mod_11(c):
    pass

### Turn 37
cs=[0,1848374,10162574,265710644,44636594]
mods=[c%11 for c in cs]
mods


### Turn 37
cs = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in cs]
print(mods)

### Turn 38
def f_using_a_half(n, max_m=100):
    best = None
    best_m=None
    for m in range(1, max_m+1):
        if m==2:  # would give b=N/2 same as a
            continue
        denom = 3*m + 2
        numer = 2 * n * m
        if numer % denom != 0:
            continue
        N = numer // denom
        if N < math.ceil(n/3):
            continue
        # verify that N works: we have a=N/2 (must be int) and b=N/m (must be int)
        if N % 2 != 0:
            continue
        if N % m != 0:
            continue
        a = N//2
        b = N//m
        if a==b or a==N or b==N:
            continue
        # check sum
        if a+b+N != n:
            continue
        # we have a valid triple, keep minimal N
        if best is None or N < best:
            best = N
            best_m=m
    return best, best_m

# test for n up to 200
errors=[]
for n in range(6,201):
    real=f_brutal_by_N(n)
    est,_=f_using_a_half(n, max_m=50)
    if est != real:
        errors.append((n,real,est))
len(errors)


### Turn 38
def f_using_a_half(n, max_m=100):
    best = None
    best_m = None
    for m in range(1, max_m + 1):
        if m == 2:
            continue
        denom = 3 * m + 2
        numer = 2 * n * m
        if numer % denom != 0:
            continue
        N = numer // denom
        if N < math.ceil(n / 3):
            continue
        if N % 2 != 0:
            continue
        if N % m != 0:
            continue
        a = N // 2
        b = N // m
        if a == b or a == N or b == N:
            continue
        if a + b + N != n:
            continue
        if best is None or N < best:
            best = N
            best_m = m
    return (best, best_m)
errors = []
for n in range(6, 201):
    real = f_brutal_by_N(n)
    est, _ = f_using_a_half(n, max_m=50)
    if est != real:
        errors.append((n, real, est))
print(len(errors))

### Turn 39
errors[:10]


### Turn 39
print(head(errors, 10))

### Turn 40
import math

K = math.factorial(2025)  # 2025!
def compute_f_n(n_c):
    # n = M + c (c could be expression involving M)
    # We'll handle c as either integer or multiple of M (coeff)
    pass

### Turn 40
import math
K = math.factorial(2025)

def compute_f_n(n_c):
    pass

### Turn 41
import math
K = math.factorial(2025)
def min_m_for_n(c_type, c_val):
    """
    c_type=0: n = M + c_val (c_val integer)
    c_type=1: n = (c_val+1)*M where c_val is extra multiplier (e.g., 4 for 4M)
    Returns tuple (m, N_modM?) not needed.
    """
    for m in range(3, 5000):
        if m == 2:
            continue
        mod = 3*m + 2
        M_mod = pow(3, K, mod)
        if c_type == 0:
            n_mod = (M_mod + (c_val % mod)) % mod
        else:
            n_mod = ((c_val + 1) * M_mod) % mod
        if (2 * n_mod * m) % mod == 0:
            return m
    return None

# compute for each case
cases = [(0,0), (1,4), (0,1848374), (0,10162574), (0,265710644), (0,44636594)]
for ct,cv in cases:
    m = min_m_for_n(ct, cv)
    print(ct,cv,m)


### Turn 41
import math
K = math.factorial(2025)

def min_m_for_n(c_type, c_val):
    """
    c_type=0: n = M + c_val (c_val integer)
    c_type=1: n = (c_val+1)*M where c_val is extra multiplier (e.g., 4 for 4M)
    Returns tuple (m, N_modM?) not needed.
    """
    for m in range(3, 5000):
        if m == 2:
            continue
        mod = 3 * m + 2
        M_mod = pow(3, K, mod)
        if c_type == 0:
            n_mod = (M_mod + c_val % mod) % mod
        else:
            n_mod = (c_val + 1) * M_mod % mod
        if 2 * n_mod * m % mod == 0:
            return m
    return None
cases = [(0, 0), (1, 4), (0, 1848374), (0, 10162574), (0, 265710644), (0, 44636594)]
for ct, cv in cases:
    m = min_m_for_n(ct, cv)
    print(ct, cv, m)

### Turn 42
def f_via_pq(n, max_p=20):
    best = None
    best_pair=None
    for p in range(2, max_p+1):
        for q in range(p+1, max_p+1):
            D = p*q + p + q
            if (n * p * q) % D != 0:
                continue
            N = n * p * q // D
            # check N divisible by p and q
            if N % p != 0 or N % q != 0:
                continue
            # ensure N >= max(p,q) (implied)
            # compute a=N//p, b=N//q
            a = N // p
            b = N // q
            if a == b or a==N or b==N or a==0 or b==0:
                continue
            # Check sum
            if a + b + N != n:
                continue
            if best is None or N < best:
                best = N
                best_pair=(p,q,a,b)
    return best, best_pair

def test_f_pq(limit=200, max_p=20):
    mism=[]
    for n in range(6, limit+1):
        real = f_brutal_by_N(n)
        est,_ = f_via_pq(n, max_p)
        if est != real:
            mism.append((n,real,est))
    return mism

mism = test_f_pq(200, max_p=20)
len(mism), mism[:20]


### Turn 42
def f_via_pq(n, max_p=20):
    best = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(p + 1, max_p + 1):
            D = p * q + p + q
            if n * p * q % D != 0:
                continue
            N = n * p * q // D
            if N % p != 0 or N % q != 0:
                continue
            a = N // p
            b = N // q
            if a == b or a == N or b == N or (a == 0) or (b == 0):
                continue
            if a + b + N != n:
                continue
            if best is None or N < best:
                best = N
                best_pair = (p, q, a, b)
    return (best, best_pair)

def test_f_pq(limit=200, max_p=20):
    mism = []
    for n in range(6, limit + 1):
        real = f_brutal_by_N(n)
        est, _ = f_via_pq(n, max_p)
        if est != real:
            mism.append((n, real, est))
    return mism
mism = test_f_pq(200, max_p=20)
print((len(mism), head(mism, 20)))

### Turn 43
mism = test_f_pq(200, max_p=50)
len(mism)


### Turn 43
mism = test_f_pq(200, max_p=50)
print(len(mism))

### Turn 44
mism[:20]


### Turn 44
print(head(mism, 20))

### Turn 45
def find_min_N_modular(c_type, c_val, max_q=2000):
    # returns (N as rational expression of M, i.e., coeff * M + const) maybe via tuple (coeff_M, const)
    # For simplicity, we will compute N as fraction of n: N = n * p * q / (p*q + p + q)
    # We'll compute the pair (p,q) that yields minimal N (as rational times n) and also compute N/M ratio as rational.
    best_ratio = None  # (num, den) representing N/n ratio simplified
    best_pair = None
    # compute n modulo denominators as needed inside loop
    for p in [2,3,4,5]:
        for q in range(p+1, max_q+1):
            D = p*q + p + q
            # condition: (n * p * q) % D ==0
            # compute n_mod for this D
            mod = D
            # compute M_mod for mod
            M_mod = pow(3, K, mod)
            if c_type == 0:
                n_mod = (M_mod + (c_val % mod)) % mod
            else:
                n_mod = ((c_val + 1) * M_mod) % mod
            if (n_mod * p * q) % mod != 0:
                continue
            # N = n * p * q / D
            # N/n ratio = p*q / D
            num = p*q
            den = D
            # reduce fraction
            g=math.gcd(num,den)
            num//=g; den//=g
            # compare ratio value (as float) maybe to find minimal N (i.e., minimal ratio)
            # Since N = ratio * n, smaller ratio => smaller N
            # For same n, ratio comparison independent of n.
            if best_ratio is None or num*best_ratio[1] < best_ratio[0]*den:
                best_ratio=(num,den)
                best_pair=(p,q)
                # If ratio already minimal possible? The minimal possible ratio is achieved by (2,3) with ratio = 6/11 ≈0.545.
                # If we find that ratio, break early maybe.
                # But we may not find it for given n.
    return best_ratio, best_pair

# test for a small n to see if we get correct f/n ratio matches brute?
def test_small_n(limit=100):
    mism=[]
    for n in range(6, limit+1):
        real=f_brutal_by_N(n)
        ratio, pair = find_min_N_modular(0,0)  # Not using n-specific; need to use actual n (can't)
    # Can't test this function without n specific; need to pass actual n mod each D.


### Turn 45
def find_min_N_modular(c_type, c_val, max_q=2000):
    best_ratio = None
    best_pair = None
    for p in [2, 3, 4, 5]:
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            if c_type == 0:
                n_mod = (M_mod + c_val % mod) % mod
            else:
                n_mod = (c_val + 1) * M_mod % mod
            if n_mod * p * q % mod != 0:
                continue
            num = p * q
            den = D
            g = math.gcd(num, den)
            num //= g
            den //= g
            if best_ratio is None or num * best_ratio[1] < best_ratio[0] * den:
                best_ratio = (num, den)
                best_pair = (p, q)
    return (best_ratio, best_pair)

def test_small_n(limit=100):
    mism = []
    for n in range(6, limit + 1):
        real = f_brutal_by_N(n)
        ratio, pair = find_min_N_modular(0, 0)

### Turn 46
def find_best_N_for_n(c_type, c_val, max_q=5000):
    # returns (ratio numerator, denominator), (p,q) yielding minimal N
    best_num = None
    best_den = None
    best_pq = None
    for p in [2,3,4,5,6,7,8,9,10]:
        for q in range(p+1, max_q+1):
            D = p*q + p + q  # denominator
            # condition for integer N: (n * p * q) % D == 0
            mod = D
            M_mod = pow(3, K, mod)
            if c_type == 0:
                n_mod = (M_mod + (c_val % mod)) % mod
            else:
                n_mod = ((c_val + 1) * M_mod) % mod
            if (n_mod * p * q) % mod != 0:
                continue
            # N ratio = p*q / D
            num = p*q
            den = D
            # reduce
            g = math.gcd(num, den)
            num //= g
            den //= g
            # compare ratio num/den (smaller is better)
            if best_num is None or num * best_den < best_num * den:
                best_num = num
                best_den = den
                best_pq = (p,q)
                # early exit if we found the theoretical minimum ratio 6/11?
                # Minimal possible ratio is 6/11 ≈0.545 for (2,3). If best ratio equals that, we could break loops.
                if num * 1 == 6 and den == 11:
                    return best_num, best_den, best_pq
    return best_num, best_den, best_pq

# Run for each case
cases = [(0,0), (1,4), (0,1848374), (0,10162574), (0,265710644), (0,44636594)]
results={}
for i,(ctype,cval) in enumerate(cases):
    num,den,pq = find_best_N_for_n(ctype,cval, max_q=500)
    results[i]=(num,den,pq)
    print(i, num, den, pq)


### Turn 46
def find_best_N_for_n(c_type, c_val, max_q=5000):
    best_num = None
    best_den = None
    best_pq = None
    for p in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            if c_type == 0:
                n_mod = (M_mod + c_val % mod) % mod
            else:
                n_mod = (c_val + 1) * M_mod % mod
            if n_mod * p * q % mod != 0:
                continue
            num = p * q
            den = D
            g = math.gcd(num, den)
            num //= g
            den //= g
            if best_num is None or num * best_den < best_num * den:
                best_num = num
                best_den = den
                best_pq = (p, q)
                if num * 1 == 6 and den == 11:
                    return (best_num, best_den, best_pq)
    return (best_num, best_den, best_pq)
cases = [(0, 0), (1, 4), (0, 1848374), (0, 10162574), (0, 265710644), (0, 44636594)]
results = {}
for i, (ctype, cval) in enumerate(cases):
    num, den, pq = find_best_N_for_n(ctype, cval, max_q=500)
    results[i] = (num, den, pq)
    print(i, num, den, pq)

### Turn 47
def find_best_N_for_n(c_type, c_val, max_q=200):
    best_num = None
    best_den = None
    best_pq = None
    for p in [2,3]:
        for q in range(p+1, max_q+1):
            D = p*q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = ((c_val+1)*M_mod) % mod if c_type==1 else (M_mod + (c_val % mod)) % mod
            if (n_mod * p * q) % mod != 0:
                continue
            num = p*q
            den = D
            g=math.gcd(num,den)
            num//=g; den//=g
            if best_num is None or num*best_den < best_num*den:
                best_num, best_den, best_pq = num, den, (p,q)
                # Early break if ratio equals lower bound for p=2 (6/11) and p=2 q=3 case
                if p==2 and q==3:
                    return best_num,best_den,best_pq
    return best_num,best_den,best_pq

for idx,(ctype,cval) in enumerate(cases):
    res = find_best_N_for_n(ctype,cval, max_q=200)
    print(idx, res)


### Turn 47
def find_best_N_for_n(c_type, c_val, max_q=200):
    best_num = None
    best_den = None
    best_pq = None
    for p in [2, 3]:
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = (c_val + 1) * M_mod % mod if c_type == 1 else (M_mod + c_val % mod) % mod
            if n_mod * p * q % mod != 0:
                continue
            num = p * q
            den = D
            g = math.gcd(num, den)
            num //= g
            den //= g
            if best_num is None or num * best_den < best_num * den:
                best_num, best_den, best_pq = (num, den, (p, q))
                if p == 2 and q == 3:
                    return (best_num, best_den, best_pq)
    return (best_num, best_den, best_pq)
for idx, (ctype, cval) in enumerate(cases):
    res = find_best_N_for_n(ctype, cval, max_q=200)
    print(idx, res)

### Turn 48
def find_best_N_for_n(c_type, c_val, max_q=500):
    best_num = None
    best_den = None
    best_pair = None
    for p in [2,3,4,5]:
        for q in range(p+1, max_q+1):
            D = p*q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = ((c_val+1) * M_mod) % mod if c_type==1 else (M_mod + (c_val % mod)) % mod
            # Check integer N condition
            if (n_mod * p * q) % mod != 0:
                continue
            # Compute N ratio numerator/denominator before reduction
            num = p * q
            den = D
            # Check that N divisible by p and q (i.e., denominator after reduction still yields integer).
            # Since N = n * num / den, and we know den | n * num. We also need p|N => N % p ==0.
            # Since N = n * num / den, N % p == 0 iff (n * num / den) % p ==0.
            # Equivalent to n * num ≡ 0 (mod p*den). But easier: after computing N as rational we can check using modulus with p*den maybe.
            # Let's compute N_mod_p = (n_mod * num // den) % p ? This may not be integer.
            # Simpler: compute N as integer using Python's big integers for small q? Not feasible.
            # Alternative: Since p divides num (since num = p*q), N = n * (p*q)/den = n * p * q / den. Since we know den divides n*p*q.
            # Write N = n * (p*q/den). Let factor = (p*q)/den as reduced fraction A/B.
            # Since den may have common factors with p, but after reduction the denominator B is coprime with p maybe.
            # To ensure N divisible by p, need B divides n * (A/p?) This is messy.
            # Instead, after reduction we can compute N_ratio = num/den reduced to a/b. Then N = n * a / b. Since a and b are coprime.
            # Need b | n. Since we already ensured denominator divides n*num, after reduction b divides n maybe.
            # So N integer automatically.
            # For divisibility by p: Since a includes factor p? Let's check.
            # Because original num = p*q, after canceling gcd with den, may lose factor p. For divisibility, we need the final N to be divisible by p.
            # We can simply compute N mod p using modular arithmetic: N ≡ (n_mod * num/den) mod p.
            # Compute inv = modular inverse of den modulo p if gcd(den, p)=1. Since p is prime perhaps, but den may be multiple of p? If den divisible by p, then N integer may have p factor already.
            # Simpler: compute N_mod_p directly by using Python's pow with modulo p*den? Not trivial.
            # Instead, we can compute N as integer using Python's arbitrary big ints for these huge n? Not feasible.
            # Instead, we can check that p divides N by checking that after reduction the numerator a is multiple of p.
            g=math.gcd(num, den)
            a = num//g
            b = den//g
            # N = n * a / b. Since gcd(a,b)=1, need b | n.
            # We know b divides n because integer condition.
            # Now check if p divides N = n*a/b. Since p | a? Let's check.
            if a % p != 0:
                continue
            # Similarly check q divides N: need q | N
            if a % q != 0:
                continue
            # Accept this solution
            # Compare ratio a/b (since N/n = a/b)
            if best_num is None or a*best_den < best_num*b:
                best_num, best_den = a, b
                best_pair = (p,q)
                # early exit if ratio equals lower bound 6/11 for p=2,q=3
                if best_num*11 == best_den*6:
                    return best_num, best_den, best_pair
    return best_num, best_den, best_pair

# Test for our six cases
for idx,(ctype,cval) in enumerate(cases):
    num,den,pq = find_best_N_for_n(ctype,cval, max_q=200)
    print(idx, num, den, pq)


### Turn 48
def find_best_N_for_n(c_type, c_val, max_q=500):
    best_num = None
    best_den = None
    best_pair = None
    for p in [2, 3, 4, 5]:
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = (c_val + 1) * M_mod % mod if c_type == 1 else (M_mod + c_val % mod) % mod
            if n_mod * p * q % mod != 0:
                continue
            num = p * q
            den = D
            g = math.gcd(num, den)
            a = num // g
            b = den // g
            if a % p != 0:
                continue
            if a % q != 0:
                continue
            if best_num is None or a * best_den < best_num * b:
                best_num, best_den = (a, b)
                best_pair = (p, q)
                if best_num * 11 == best_den * 6:
                    return (best_num, best_den, best_pair)
    return (best_num, best_den, best_pair)
for idx, (ctype, cval) in enumerate(cases):
    num, den, pq = find_best_N_for_n(ctype, cval, max_q=200)
    print(idx, num, den, pq)

### Turn 49
def test_pair(c_type,c_val,p,q):
    D=p*q + p + q
    mod=D
    M_mod=pow(3, K, mod)
    n_mod = ((c_val+1)*M_mod)%mod if c_type==1 else (M_mod + (c_val%mod))%mod
    ok = (n_mod * p * q) % mod == 0
    g=math.gcd(p*q, D)
    a=p*q//g
    b=D//g
    # check divisibility b divides n (i.e., n % b==0)
    # compute n % b via modular exponent
    M_mod_b = pow(3, K, b)
    n_mod_b = ((c_val+1)*M_mod_b)%b if c_type==1 else (M_mod_b + (c_val % b))%b
    ok2 = n_mod_b == 0
    # N = (n//b) * a ; compute N mod p maybe:
    # Compute n // b mod p using n_mod (need division). Since b is small, we can compute n // b exactly as integer using big int? Not possible.
    return ok, ok2, D, g, a, b

print(test_pair(1,4,3,6))


### Turn 49
def test_pair(c_type, c_val, p, q):
    D = p * q + p + q
    mod = D
    M_mod = pow(3, K, mod)
    n_mod = (c_val + 1) * M_mod % mod if c_type == 1 else (M_mod + c_val % mod) % mod
    ok = n_mod * p * q % mod == 0
    g = math.gcd(p * q, D)
    a = p * q // g
    b = D // g
    M_mod_b = pow(3, K, b)
    n_mod_b = (c_val + 1) * M_mod_b % b if c_type == 1 else (M_mod_b + c_val % b) % b
    ok2 = n_mod_b == 0
    return (ok, ok2, D, g, a, b)
print(test_pair(1, 4, 3, 6))

### Turn 50
def find_best_debug(c_type,c_val,max_q=20):
    best_num=None; best_den=None; best_pair=None
    for p in [2,3,4,5]:
        for q in range(p+1, max_q+1):
            D = p*q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = ((c_val+1)*M_mod)%mod if c_type==1 else (M_mod + (c_val%mod))%mod
            if (n_mod * p * q) % mod !=0:
                continue
            # compute a,b
            g=math.gcd(p*q, D)
            a = (p*q)//g
            b = D//g
            # check b divides n
            M_mod_b = pow(3, K, b)
            n_mod_b = ((c_val+1)*M_mod_b)%b if c_type==1 else (M_mod_b + (c_val % b))%b
            if n_mod_b !=0:
                continue
            # we accept; ratio = a/b
            if best_num is None or a*best_den < best_num*b:
                best_num, best_den, best_pair = a, b, (p,q)
                print("New best:", p,q,"ratio",a,b)
    return best_num, best_den, best_pair

find_best_debug(1,4, max_q=10)


### Turn 50
def find_best_debug(c_type, c_val, max_q=20):
    best_num = None
    best_den = None
    best_pair = None
    for p in [2, 3, 4, 5]:
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = (c_val + 1) * M_mod % mod if c_type == 1 else (M_mod + c_val % mod) % mod
            if n_mod * p * q % mod != 0:
                continue
            g = math.gcd(p * q, D)
            a = p * q // g
            b = D // g
            M_mod_b = pow(3, K, b)
            n_mod_b = (c_val + 1) * M_mod_b % b if c_type == 1 else (M_mod_b + c_val % b) % b
            if n_mod_b != 0:
                continue
            if best_num is None or a * best_den < best_num * b:
                best_num, best_den, best_pair = (a, b, (p, q))
                print('New best:', p, q, 'ratio', a, b)
    return (best_num, best_den, best_pair)
print(find_best_debug(1, 4, max_q=10))

### Turn 51
find_best_debug(0,0, max_q=10)


### Turn 51
print(find_best_debug(0, 0, max_q=10))

### Turn 52
def find_best_N_for_n(c_type,c_val, max_q=200):
    best_num=None; best_den=None; best_pair=None
    for p in range(2,6):
        for q in range(p+1, max_q+1):
            D = p*q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = ((c_val+1)*M_mod)%mod if c_type==1 else (M_mod + (c_val%mod))%mod
            if (n_mod * p * q) % mod != 0:
                continue
            g=math.gcd(p*q, D)
            a=(p*q)//g
            b=D//g
            # check b divides n
            M_mod_b = pow(3, K, b)
            n_mod_b = ((c_val+1)*M_mod_b)%b if c_type==1 else (M_mod_b + (c_val % b))%b
            if n_mod_b !=0:
                continue
            # ratio a/b
            if best_num is None or a*best_den < best_num*b:
                best_num,best_den,best_pair = a,b,(p,q)
    return best_num,best_den,best_pair

print(find_best_N_for_n(0,0, max_q=50))


### Turn 52
def find_best_N_for_n(c_type, c_val, max_q=200):
    best_num = None
    best_den = None
    best_pair = None
    for p in range(2, 6):
        for q in range(p + 1, max_q + 1):
            D = p * q + p + q
            mod = D
            M_mod = pow(3, K, mod)
            n_mod = (c_val + 1) * M_mod % mod if c_type == 1 else (M_mod + c_val % mod) % mod
            if n_mod * p * q % mod != 0:
                continue
            g = math.gcd(p * q, D)
            a = p * q // g
            b = D // g
            M_mod_b = pow(3, K, b)
            n_mod_b = (c_val + 1) * M_mod_b % b if c_type == 1 else (M_mod_b + c_val % b) % b
            if n_mod_b != 0:
                continue
            if best_num is None or a * best_den < best_num * b:
                best_num, best_den, best_pair = (a, b, (p, q))
    return (best_num, best_den, best_pair)
print(find_best_N_for_n(0, 0, max_q=50))

### Turn 53
cases = [(0,0), (1,4), (0,1848374), (0,10162574), (0,265710644), (0,44636594)]
def compute_best_ratio(c_type,c_val):
    best_num,best_den,best_pair = find_best_N_for_n(c_type,c_val, max_q=2000)
    return best_num,best_den,best_pair

for idx,(ct,cv) in enumerate(cases):
    num,den,pair = compute_best_ratio(ct,cv)
    print(idx, "ratio", num, "/", den, "pair", pair)


### Turn 53
cases = [(0, 0), (1, 4), (0, 1848374), (0, 10162574), (0, 265710644), (0, 44636594)]

def compute_best_ratio(c_type, c_val):
    best_num, best_den, best_pair = find_best_N_for_n(c_type, c_val, max_q=2000)
    return (best_num, best_den, best_pair)
for idx, (ct, cv) in enumerate(cases):
    num, den, pair = compute_best_ratio(ct, cv)
    print(idx, 'ratio', num, '/', den, 'pair', pair)

### Turn 54
def compute_best_ratio_limited(c_type,c_val, max_q=500):
    best_num,best_den,best_pair = find_best_N_for_n(c_type,c_val, max_q)
    return best_num,best_den,best_pair

for idx,(ct,cv) in enumerate(cases):
    num,den,pair = compute_best_ratio_limited(ct,cv, max_q=500)
    print(idx, num, den, pair)


### Turn 54
def compute_best_ratio_limited(c_type, c_val, max_q=500):
    best_num, best_den, best_pair = find_best_N_for_n(c_type, c_val, max_q)
    return (best_num, best_den, best_pair)
for idx, (ct, cv) in enumerate(cases):
    num, den, pair = compute_best_ratio_limited(ct, cv, max_q=500)
    print(idx, num, den, pair)

