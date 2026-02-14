# Auto-extracted from: P10_v9r6.md
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
def f_bruteforce(n):
    min_l = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b: 
                continue
            # distinct
            import math
            l = math.lcm(a,b,c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_bruteforce_up_to(N):
    res = {}
    for n in range(6, N+1):
        res[n] = f_bruteforce(n)
    return res

print([ (n,f_bruteforce(n)) for n in range(6,21) ])


### Turn 0
def f_bruteforce(n):
    min_l = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            import math
            l = math.lcm(a, b, c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_bruteforce_up_to(N):
    res = {}
    for n in range(6, N + 1):
        res[n] = f_bruteforce(n)
    return res
print([(n, f_bruteforce(n)) for n in range(6, 21)])

### Turn 1
vals = [(n,f_bruteforce(n)) for n in range(6,101)]
vals[:30]


### Turn 1
vals = [(n, f_bruteforce(n)) for n in range(6, 101)]
print(head(vals, 30))

### Turn 2
vals = [(n,f_bruteforce(n)) for n in range(6,201)]
max_val = max(v for _,v in vals)
max_val


### Turn 2
vals = [(n, f_bruteforce(n)) for n in range(6, 201)]
max_val = max((v for _, v in vals))
print(max_val)

### Turn 3
def ratio(a,b,c):
    import math
    return math.lcm(a,b,c)/(a+b+c)

def min_ratio(limit=20):
    best = (None,None)
    best_trip = None
    for a in range(1,limit+1):
        for b in range(a+1,limit+1):
            for c in range(b+1,limit+1):
                r = ratio(a,b,c)
                if best[0] is None or r < best[0]:
                    best = (r, (a,b,c))
    return best

best = min_ratio(30)
best


### Turn 3
def ratio(a, b, c):
    import math
    return math.lcm(a, b, c) / (a + b + c)

def min_ratio(limit=20):
    best = (None, None)
    best_trip = None
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):
            for c in range(b + 1, limit + 1):
                r = ratio(a, b, c)
                if best[0] is None or r < best[0]:
                    best = (r, (a, b, c))
    return best
best = min_ratio(30)
print(best)

### Turn 4
best = min_ratio(100)
best


### Turn 4
best = min_ratio(100)
print(best)

### Turn 5
def min_ratio_search(N):
    import math, itertools
    best = (1e9, None)
    for a in range(1,N+1):
        for b in range(a+1,N+1):
            for c in range(b+1,N+1):
                L = math.lcm(a,b,c)
                s = a+b+c
                r = L/s
                if r < best[0]:
                    best = (r, (a,b,c))
    return best

best = min_ratio_search(200)
best


### Turn 5
def min_ratio_search(N):
    import math, itertools
    best = (1000000000.0, None)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                L = math.lcm(a, b, c)
                s = a + b + c
                r = L / s
                if r < best[0]:
                    best = (r, (a, b, c))
    return best
best = min_ratio_search(200)
print(best)

### Turn 6
def f_bruteforce(n):
    import math
    best = None
    for a in range(1,n):
        for b in range(a+1,n):
            c = n-a-b
            if c <= b:
                continue
            L = math.lcm(a,b,c)
            if best is None or L < best:
                best = L
    return best

def f_list(N):
    res = {}
    for n in range(6, N+1):
        res[n] = f_bruteforce(n)
    return res

def compare(N):
    res = {}
    for n in range(6,N+1):
        res[n] = (f_bruteforce(n), (6*n)//11, math.ceil(6*n/11))
    return res

comp = compare(200)
for n in range(6,51):
    fval, floor6_11, ceil6_11 = comp[n]
    if fval not in (floor6_11, ceil6_11):
        print(n, fval, floor6_11, ceil6_11)


### Turn 6
def f_bruteforce(n):
    import math
    best = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if best is None or L < best:
                best = L
    return best

def f_list(N):
    res = {}
    for n in range(6, N + 1):
        res[n] = f_bruteforce(n)
    return res

def compare(N):
    res = {}
    for n in range(6, N + 1):
        res[n] = (f_bruteforce(n), 6 * n // 11, math.ceil(6 * n / 11))
    return res
comp = compare(200)
for n in range(6, 51):
    fval, floor6_11, ceil6_11 = comp[n]
    if fval not in (floor6_11, ceil6_11):
        print(n, fval, floor6_11, ceil6_11)

### Turn 7
for n in range(11,200,11):
    print(n, f_bruteforce(n), (6*n)//11)


### Turn 7
for n in range(11, 200, 11):
    print(n, f_bruteforce(n), 6 * n // 11)

### Turn 8
def f_bruteforce_fast(n):
    import math
    best = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if best is None or L < best:
                best = L
    return best

def diff(N):
    diffs = {}
    for n in range(6, N+1):
        diffs[n] = f_bruteforce_fast(n) - (6*n)//11
    return diffs

d = diff(200)
sorted(set(d.values()))[:20]


### Turn 8
def f_bruteforce_fast(n):
    import math
    best = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if best is None or L < best:
                best = L
    return best

def diff(N):
    diffs = {}
    for n in range(6, N + 1):
        diffs[n] = f_bruteforce_fast(n) - 6 * n // 11
    return diffs
d = diff(200)
print(head(sorted(set(d.values())), 20))

### Turn 9
from collections import defaultdict
def compute_residue_stats(N=200):
    stats = defaultdict(list)
    for n in range(6,N+1):
        diff = f_bruteforce_fast(n) - (6*n)//11
        stats[n%11].append(diff)
    return {r: (min(v), max(v)) for r,v in stats.items()}
compute_residue_stats(200)


### Turn 9
from collections import defaultdict

def compute_residue_stats(N=200):
    stats = defaultdict(list)
    for n in range(6, N + 1):
        diff = f_bruteforce_fast(n) - 6 * n // 11
        stats[n % 11].append(diff)
    return {r: (min(v), max(v)) for r, v in stats.items()}
print(compute_residue_stats(200))

### Turn 10
def f_bruteforce(n):
    import math, itertools
    best = None
    for a in range(1,n):
        for b in range(a+1,n):
            c=n-a-b
            if c<=b: continue
            L=math.lcm(a,b,c)
            if best is None or L<best:
                best=L
    return best

def diff_vals(N):
    diffs = {}
    for n in range(6,N+1):
        diff = f_bruteforce(n) - ((6*n+10)//11)  # ceil 6n/11
        diffs[n]=diff
    return diffs

diffs=diff_vals(500)
max_diff = max(diffs.values())
min_diff = min(diffs.values())
max_diff, min_diff


### Turn 10
def f_bruteforce(n):
    import math, itertools
    best = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if best is None or L < best:
                best = L
    return best

def diff_vals(N):
    diffs = {}
    for n in range(6, N + 1):
        diff = f_bruteforce(n) - (6 * n + 10) // 11
        diffs[n] = diff
    return diffs
diffs = diff_vals(500)
max_diff = max(diffs.values())
min_diff = min(diffs.values())
print((max_diff, min_diff))

### Turn 11
def f_estimate(n):
    k = n//11
    r = n%11
    d = (6*r + 10)//11  # ceil(6r/11)
    return 6*k + d

def test(N=200):
    bad=[]
    for n in range(6,N+1):
        f_true=f_bruteforce(n)
        f_est=f_estimate(n)
        if f_true!=f_est:
            bad.append((n,f_true,f_est))
    return bad

bad=test(300)
len(bad), bad[:20]


### Turn 11
def f_estimate(n):
    k = n // 11
    r = n % 11
    d = (6 * r + 10) // 11
    return 6 * k + d

def test(N=200):
    bad = []
    for n in range(6, N + 1):
        f_true = f_bruteforce(n)
        f_est = f_estimate(n)
        if f_true != f_est:
            bad.append((n, f_true, f_est))
    return bad
bad = test(300)
print((len(bad), head(bad, 20)))

### Turn 12
def brute_f_up_to(N):
    import math
    f = [0]*(N+1)
    for n in range(6,N+1):
        best = None
        for a in range(1,n):
            for b in range(a+1,n):
                c = n - a - b
                if c <= b: continue
                L = math.lcm(a,b,c)
                if best is None or L < best:
                    best = L
        f[n] = best
    return f

N=500
fvals = brute_f_up_to(N)

def check_formula(N):
    mismatches=[]
    for n in range(6,N+1):
        k=n//11
        r=n%11
        d=(6*r+10)//11
        est=6*k+d
        if fvals[n]!=est:
            mismatches.append((n,fvals[n],est))
    return mismatches

mism=check_formula(500)
len(mism), mism[:10]


### Turn 12
def brute_f_up_to(N):
    import math
    f = [0] * (N + 1)
    for n in range(6, N + 1):
        best = None
        for a in range(1, n):
            for b in range(a + 1, n):
                c = n - a - b
                if c <= b:
                    continue
                L = math.lcm(a, b, c)
                if best is None or L < best:
                    best = L
        f[n] = best
    return f
N = 500
fvals = brute_f_up_to(N)

def check_formula(N):
    mismatches = []
    for n in range(6, N + 1):
        k = n // 11
        r = n % 11
        d = (6 * r + 10) // 11
        est = 6 * k + d
        if fvals[n] != est:
            mismatches.append((n, fvals[n], est))
    return mismatches
mism = check_formula(500)
print((len(mism), head(mism, 10)))

### Turn 13
def f_est_bruteforce_large(N):
    import math
    # naive but for moderate N like 11000 perhaps fine.
    return f_bruteforce(N)

def f_via_L(N, limitL=None):
    # Search L increasing from 1 upwards; find if sum N can be achieved.
    import math, itertools
    # We can precompute divisors of L and check combos.
    L=1
    while True:
        # compute divisors
        divs = []
        i=1
        while i*i <= L:
            if L % i == 0:
                divs.append(i)
                if i*i!=L:
                    divs.append(L//i)
            i+=1
        divs.sort()
        if len(divs) >= 3:
            # check any 3 distinct divisors sum to N
            # use combination enumeration for small number of divisors
            from itertools import combinations
            for a,b,c in combinations(divs,3):
                if a+b+c==N:
                    return L
        L+=1

# Let's test for n = 11*k + r for k big, r various.
def f_min_L_est(k, r):
    n = 11*k + r
    # search L from 6*k up to 6*k+30 maybe
    import math, itertools
    # compute divisors for each candidate L
    for d in range(0,20):
        L = 6*k + d
        # compute divisors of L
        divs=[]
        i=1
        while i*i<=L:
            if L%i==0:
                divs.append(i)
                if i*i!=L:
                    divs.append(L//i)
            i+=1
        divs.sort()
        if len(divs)>=3:
            # check for sum N
            for a,b,c in itertools.combinations(divs,3):
                if a+b+c==n:
                    return L
    return None

def test_large_k(k):
    res={}
    for r in range(11):
        L = f_min_L_est(k,r)
        res[r]=L
    return res

print(test_large_k(1000))


### Turn 13
def f_est_bruteforce_large(N):
    import math
    return f_bruteforce(N)

def f_via_L(N, limitL=None):
    import math, itertools
    L = 1
    while True:
        divs = []
        i = 1
        while i * i <= L:
            if L % i == 0:
                divs.append(i)
                if i * i != L:
                    divs.append(L // i)
            i += 1
        divs.sort()
        if len(divs) >= 3:
            from itertools import combinations
            for a, b, c in combinations(divs, 3):
                if a + b + c == N:
                    return L
        L += 1

def f_min_L_est(k, r):
    n = 11 * k + r
    import math, itertools
    for d in range(0, 20):
        L = 6 * k + d
        divs = []
        i = 1
        while i * i <= L:
            if L % i == 0:
                divs.append(i)
                if i * i != L:
                    divs.append(L // i)
            i += 1
        divs.sort()
        if len(divs) >= 3:
            for a, b, c in itertools.combinations(divs, 3):
                if a + b + c == n:
                    return L
    return None

def test_large_k(k):
    res = {}
    for r in range(11):
        L = f_min_L_est(k, r)
        res[r] = L
    return res
print(test_large_k(1000))

### Turn 14
def f_min_L_est_range(k, maxd=200):
    res={}
    for r in range(11):
        target=11*k+r
        found=None
        for d in range(0,maxd+1):
            L=6*k+d
            # compute divisors list quickly using sympy maybe
            import sympy as sp
            divs = sp.divisors(L)
            # ensure at least 3 distinct divisors
            if len(divs) < 3:
                continue
            # check combos - maybe large but we can break early if sum too big.
            # Instead we can search for two divisors a,b such that L + a + b = target, i.e., a+b = target - L
            need = target - L
            if need <= 0:
                continue
            # need two distinct proper divisors summing to need.
            # iterate over divisors (exclude L)
            proper = [d for d in divs if d != L]
            # Use set for fast lookup
            sset=set(proper)
            for a in proper:
                b = need - a
                if b> a and b in sset:
                    found = L
                    break
            if found is not None:
                break
        res[r]=found
    return res

print(f_min_L_est_range(1000, maxd=200))


### Turn 14
def f_min_L_est_range(k, maxd=200):
    res = {}
    for r in range(11):
        target = 11 * k + r
        found = None
        for d in range(0, maxd + 1):
            L = 6 * k + d
            import sympy as sp
            divs = sp.divisors(L)
            if len(divs) < 3:
                continue
            need = target - L
            if need <= 0:
                continue
            proper = [d for d in divs if d != L]
            sset = set(proper)
            for a in proper:
                b = need - a
                if b > a and b in sset:
                    found = L
                    break
            if found is not None:
                break
        res[r] = found
    return res
print(f_min_L_est_range(1000, maxd=200))

### Turn 15
def find_L_for_residue(k,r,maxd=2000):
    target = 11*k+r
    for d in range(0,maxd+1):
        L = 6*k + d
        need = target - L
        if need <= 0: continue
        # need two distinct proper divisors summing to need
        divs = sympy.divisors(L)
        proper = [d_ for d_ in divs if d_ != L]
        sset = set(proper)
        for a in proper:
            b = need - a
            if b > a and b in sset:
                return L, a, b
    return None

import sympy, math, itertools, sys, time, math, collections, random
print(find_L_for_residue(1000,1,1000))


### Turn 15
def find_L_for_residue(k, r, maxd=2000):
    target = 11 * k + r
    for d in range(0, maxd + 1):
        L = 6 * k + d
        need = target - L
        if need <= 0:
            continue
        divs = sympy.divisors(L)
        proper = [d_ for d_ in divs if d_ != L]
        sset = set(proper)
        for a in proper:
            b = need - a
            if b > a and b in sset:
                return (L, a, b)
    return None
import sympy, math, itertools, sys, time, math, collections, random
print(find_L_for_residue(1000, 1, 1000))

### Turn 16
def min_L_for_k_r(k,r,limit=2000):
    target=11*k+r
    best=None
    best_details=None
    for L in range(6*k,6*k+limit+1):
        need=target-L
        if need<=0: continue
        divs=sympy.divisors(L)
        if len(divs)<3: continue
        proper=set(d for d in divs if d!=L)
        for a in proper:
            b=need-a
            if b>a and b in proper:
                best=L
                best_details=(a,b)
                return best,best_details
    return None

print(min_L_for_k_r(1000,1,2000))


### Turn 16
def min_L_for_k_r(k, r, limit=2000):
    target = 11 * k + r
    best = None
    best_details = None
    for L in range(6 * k, 6 * k + limit + 1):
        need = target - L
        if need <= 0:
            continue
        divs = sympy.divisors(L)
        if len(divs) < 3:
            continue
        proper = set((d for d in divs if d != L))
        for a in proper:
            b = need - a
            if b > a and b in proper:
                best = L
                best_details = (a, b)
                return (best, best_details)
    return None
print(min_L_for_k_r(1000, 1, 2000))

### Turn 17
c_values = [0, 4*3**0, 1848374, 10162574, 265710644, 44636594]
mods = [c%11 for c in c_values]
mods


### Turn 17
c_values = [0, 4 * 3 ** 0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_values]
print(mods)

### Turn 18
1848374 % 11


### Turn 18
print(1848374 % 11)

### Turn 19
10162574 % 11


### Turn 19
print(10162574 % 11)

### Turn 20
265710644 % 11


### Turn 20
print(265710644 % 11)

### Turn 21
44636594 % 11


### Turn 21
print(44636594 % 11)

### Turn 22
def min_ratio_nonzero(limit=100):
    import math
    best = (1, None)
    for x in range(2,limit+1):
        for y in range(2,limit+1):
            if x == y:
                continue
            denom = x*y + x + y
            if denom % 11 == 0:
                continue
            ratio = (x*y)/denom
            if ratio < best[0]:
                best = (ratio, (x,y,denom))
    return best

best= min_ratio_nonzero(200)
best


### Turn 22
def min_ratio_nonzero(limit=100):
    import math
    best = (1, None)
    for x in range(2, limit + 1):
        for y in range(2, limit + 1):
            if x == y:
                continue
            denom = x * y + x + y
            if denom % 11 == 0:
                continue
            ratio = x * y / denom
            if ratio < best[0]:
                best = (ratio, (x, y, denom))
    return best
best = min_ratio_nonzero(200)
print(best)

### Turn 23
def min_ratio_nonzero_up_to(limit):
    best = (1.0, None)
    for x in range(2, limit+1):
        for y in range(2, limit+1):
            if x == y:
                continue
            denom = x*y + x + y
            if denom % 11 == 0:
                continue
            ratio = (x*y)/denom
            if ratio < best[0]:
                best = (ratio, (x,y,denom))
    return best

best = min_ratio_nonzero_up_to(1000)
best


### Turn 23
def min_ratio_nonzero_up_to(limit):
    best = (1.0, None)
    for x in range(2, limit + 1):
        for y in range(2, limit + 1):
            if x == y:
                continue
            denom = x * y + x + y
            if denom % 11 == 0:
                continue
            ratio = x * y / denom
            if ratio < best[0]:
                best = (ratio, (x, y, denom))
    return best
best = min_ratio_nonzero_up_to(1000)
print(best)

### Turn 24
import math, sympy, itertools, sys, time, functools, collections, math, random, sys, math

# compute t = 2025!
t = math.factorial(2025)
t_bitlen = t.bit_length()
print("t bits:", t_bitlen)


### Turn 24
import math, sympy, itertools, sys, time, functools, collections, math, random, sys, math
t = math.factorial(2025)
t_bitlen = t.bit_length()
print('t bits:', t_bitlen)

### Turn 25
def minimal_ratio_for_N(c_val, include_4M=False):
    # c_val is integer offset; if include_4M True, then N = 5M (since c = 4M)
    # returns minimal ratio as Fraction and the s that achieves it
    from fractions import Fraction
    # Generate candidate pairs up to some bound
    best_ratio = Fraction(1,1)  # start high
    best_pair = None
    # We'll search x up to, say, 100 maybe; if not found, expand.
    max_x = 100
    for x in range(2, max_x+1):
        for y in range(2, max_x+1):
            if x == y:
                continue
            s = x*y + x + y
            # compute N mod s
            # M_mod_s
            M_mod_s = pow(3, t, s)  # M = 3^t
            if include_4M:
                # N = 5M, c offset not used
                N_mod_s = (5 * M_mod_s) % s
            else:
                N_mod_s = (M_mod_s + (c_val % s)) % s
            if N_mod_s != 0:
                continue
            # s divides N
            ratio = Fraction(x*y, s)
            if ratio < best_ratio:
                best_ratio = ratio
                best_pair = (x,y,s)
    return best_ratio, best_pair

# Define the c values
c_list = [0, None, 1848374, 10162574, 265710644, 44636594]  # placeholder for 4M case as None
ratios = []
for idx, c in enumerate(c_list):
    if idx == 1:  # c = 4M case
        ratio, pair = minimal_ratio_for_N(0, include_4M=True)
    else:
        ratio, pair = minimal_ratio_for_N(c, include_4M=False)
    ratios.append((ratio, pair))
ratios


### Turn 25
def minimal_ratio_for_N(c_val, include_4M=False):
    from fractions import Fraction
    best_ratio = Fraction(1, 1)
    best_pair = None
    max_x = 100
    for x in range(2, max_x + 1):
        for y in range(2, max_x + 1):
            if x == y:
                continue
            s = x * y + x + y
            M_mod_s = pow(3, t, s)
            if include_4M:
                N_mod_s = 5 * M_mod_s % s
            else:
                N_mod_s = (M_mod_s + c_val % s) % s
            if N_mod_s != 0:
                continue
            ratio = Fraction(x * y, s)
            if ratio < best_ratio:
                best_ratio = ratio
                best_pair = (x, y, s)
    return (best_ratio, best_pair)
c_list = [0, None, 1848374, 10162574, 265710644, 44636594]
ratios = []
for idx, c in enumerate(c_list):
    if idx == 1:
        ratio, pair = minimal_ratio_for_N(0, include_4M=True)
    else:
        ratio, pair = minimal_ratio_for_N(c, include_4M=False)
    ratios.append((ratio, pair))
print(ratios)

### Turn 26
import math
from fractions import Fraction

t = math.factorial(2025)  # big integer

# candidate N list identifiers
c_vals = [0, "4M", 1848374, 10162574, 265710644, 44636594]

# We'll compute minimal ratio for each
# We'll generate candidate pairs up to max_x=200 initially
max_x = 200

# Precompute list of pairs and their s and ratio
pairs = []
for x in range(2, max_x+1):
    for y in range(2, max_x+1):
        if x == y:
            continue
        s = x*y + x + y
        ratio = Fraction(x*y, s)
        pairs.append((ratio, x, y, s))
# sort by ratio ascending
pairs.sort(key=lambda tup: tup[0])

# Determine max s we need to precompute M mod s
max_s = max(p[3] for p in pairs)
max_s


### Turn 26
import math
from fractions import Fraction
t = math.factorial(2025)
c_vals = [0, '4M', 1848374, 10162574, 265710644, 44636594]
max_x = 200
pairs = []
for x in range(2, max_x + 1):
    for y in range(2, max_x + 1):
        if x == y:
            continue
        s = x * y + x + y
        ratio = Fraction(x * y, s)
        pairs.append((ratio, x, y, s))
pairs.sort(key=lambda tup: tup[0])
max_s = max((p[3] for p in pairs))
print(max_s)

### Turn 27
def best_triple(n):
    import math
    bestL = None
    bestTrip = None
    for a in range(1,n):
        for b in range(a+1,n):
            c = n-a-b
            if c <= b: continue
            L = math.lcm(a,b,c)
            if bestL is None or L < bestL:
                bestL = L
                bestTrip = (a,b,c)
    return bestL, bestTrip

print(best_triple(8))


### Turn 27
def best_triple(n):
    import math
    bestL = None
    bestTrip = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            L = math.lcm(a, b, c)
            if bestL is None or L < bestL:
                bestL = L
                bestTrip = (a, b, c)
    return (bestL, bestTrip)
print(best_triple(8))

### Turn 28
import math
from fractions import Fraction

t = math.factorial(2025)

# N definitions: store as type: (type, c) where type indicates normal (offset) or multiple of M (factor)
N_info = [
    ("offset", 0),
    ("multiple", 5),  # 5M
    ("offset", 1848374),
    ("offset", 10162574),
    ("offset", 265710644),
    ("offset", 44636594)
]

# generate candidate pairs
max_x = 200
candidates = []
for x in range(2, max_x+1):
    for y in range(2, max_x+1):
        if x == y: continue
        s = x*y + x + y
        ratio = Fraction(x*y, s)  # xy/s
        candidates.append((ratio, x, y, s))
# sort by ratio ascending
candidates.sort(key=lambda tup: tup[0])

# Precompute M_mod_s for each distinct s encountered
unique_s = sorted({cand[3] for cand in candidates})
M_mod = {}
for s in unique_s:
    M_mod[s] = pow(3, t, s)

def find_best_ratio(N_type, param):
    # return best ratio and corresponding s
    # N_type == "offset": N = M + param
    # N_type == "multiple": N = param * M, where param is integer factor
    best = Fraction(10**9,1)  # large
    best_cand = None
    for ratio, x, y, s in candidates:
        # check divisibility
        M_mod_s = M_mod[s]
        if N_type == "offset":
            N_mod_s = (M_mod_s + (param % s)) % s
        else:  # multiple
            factor = param % s
            N_mod_s = (M_mod_s * factor) % s
        if N_mod_s != 0:
            continue
        # s divides N, so ratio feasible
        if ratio < best:
            best = ratio
            best_cand = (x, y, s)
            # Since candidates sorted by ratio, we can break early when first found?
            # But need to ensure that earlier ratio may not divide N
            # Continue until we find first dividing; that's minimal.
            break
    return best, best_cand

results = []
for info in N_info:
    typ, param = info
    best_ratio, cand = find_best_ratio(typ, param)
    results.append((best_ratio, cand))
results


### Turn 28
import math
from fractions import Fraction
t = math.factorial(2025)
N_info = [('offset', 0), ('multiple', 5), ('offset', 1848374), ('offset', 10162574), ('offset', 265710644), ('offset', 44636594)]
max_x = 200
candidates = []
for x in range(2, max_x + 1):
    for y in range(2, max_x + 1):
        if x == y:
            continue
        s = x * y + x + y
        ratio = Fraction(x * y, s)
        candidates.append((ratio, x, y, s))
candidates.sort(key=lambda tup: tup[0])
unique_s = sorted({cand[3] for cand in candidates})
M_mod = {}
for s in unique_s:
    M_mod[s] = pow(3, t, s)

def find_best_ratio(N_type, param):
    best = Fraction(10 ** 9, 1)
    best_cand = None
    for ratio, x, y, s in candidates:
        M_mod_s = M_mod[s]
        if N_type == 'offset':
            N_mod_s = (M_mod_s + param % s) % s
        else:
            factor = param % s
            N_mod_s = M_mod_s * factor % s
        if N_mod_s != 0:
            continue
        if ratio < best:
            best = ratio
            best_cand = (x, y, s)
            break
    return (best, best_cand)
results = []
for info in N_info:
    typ, param = info
    best_ratio, cand = find_best_ratio(typ, param)
    results.append((best_ratio, cand))
print(results)

### Turn 29
def find_best_ratio_power3(max_e=10):
    import math
    best_ratio = Fraction(10,1)  # big
    best_info = None
    for e in range(2, max_e+1):  # e>=2 because s must be >? but we start from 2 (s=9)
        s = 3**e
        # Solve xy + x + y = s => (x+1)(y+1) = s + 1
        target = s + 1
        # factor target to find integer pairs x+1, y+1
        # We need distinct x,y>1 => x+1>2, y+1>2, and distinct
        # iterate over divisors
        import sympy as sp
        for d in sp.divisors(target):
            e2 = target // d
            if d <= e2: continue  # ensure distinct (order)
            x = d - 1
            y = e2 - 1
            if x <= 1 or y <= 1 or x == y:
                continue
            ratio = Fraction(x*y, s)
            if ratio < best_ratio:
                best_ratio = ratio
                best_info = (e, s, x, y, ratio)
    return best_ratio, best_info

best_ratio, info = find_best_ratio_power3(12)
best_ratio, info


### Turn 29
def find_best_ratio_power3(max_e=10):
    import math
    best_ratio = Fraction(10, 1)
    best_info = None
    for e in range(2, max_e + 1):
        s = 3 ** e
        target = s + 1
        import sympy as sp
        for d in sp.divisors(target):
            e2 = target // d
            if d <= e2:
                continue
            x = d - 1
            y = e2 - 1
            if x <= 1 or y <= 1 or x == y:
                continue
            ratio = Fraction(x * y, s)
            if ratio < best_ratio:
                best_ratio = ratio
                best_info = (e, s, x, y, ratio)
    return (best_ratio, best_info)
best_ratio, info = find_best_ratio_power3(12)
print((best_ratio, info))

### Turn 30
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(34,53) + Fraction(30,47) + Fraction(130,197) + Fraction(110,167)
S


### Turn 30
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(34, 53) + Fraction(30, 47) + Fraction(130, 197) + Fraction(110, 167)
print(S)

### Turn 31
import math
math.gcd(540747228,81951409)


### Turn 31
import math
print(math.gcd(540747228, 81951409))

### Turn 32
p=540747228
q=81951409
p+q


### Turn 32
p = 540747228
q = 81951409
print(p + q)

### Turn 33
(p+q) % 99991


### Turn 33
print((p + q) % 99991)

### Turn 34
def min_ratio_for_s(s):
    # find min xy/s for any distinct x,y>1 solving xy + x + y = s
    import sympy as sp
    best = None
    # x+1, y+1 are divisors of s+1
    target = s + 1
    for d in sp.divisors(target):
        e = target // d
        if d <= e: continue
        x = d - 1
        y = e - 1
        if x <= 1 or y <= 1 or x == y:
            continue
        ratio = Fraction(x*y, s)
        if best is None or ratio < best:
            best = ratio
    return best

ratios = {}
for s in range(12, 300):  # start from maybe 12 (since minimal s for x>1,y>1 is 12? Actually smallest s from pair (2,3)=11, but we exclude s=11 because that's 6/11.
    r = min_ratio_for_s(s)
    if r is not None:
        ratios[s] = r

# sort by ratio
sorted_ratios = sorted(ratios.items(), key=lambda kv: kv[1])
sorted_ratios[:20]


### Turn 34
def min_ratio_for_s(s):
    import sympy as sp
    best = None
    target = s + 1
    for d in sp.divisors(target):
        e = target // d
        if d <= e:
            continue
        x = d - 1
        y = e - 1
        if x <= 1 or y <= 1 or x == y:
            continue
        ratio = Fraction(x * y, s)
        if best is None or ratio < best:
            best = ratio
    return best
ratios = {}
for s in range(12, 300):
    r = min_ratio_for_s(s)
    if r is not None:
        ratios[s] = r
sorted_ratios = sorted(ratios.items(), key=lambda kv: kv[1])
print(head(sorted_ratios, 20))

### Turn 35
def solution_for_s(s):
    import sympy as sp
    target = s+1
    sols=[]
    for d in sp.divisors(target):
        e = target//d
        if d<=e: continue
        x=d-1
        y=e-1
        if x>1 and y>1 and x!=y:
            sols.append((x,y))
    return sols

solution_for_s(50)


### Turn 35
def solution_for_s(s):
    import sympy as sp
    target = s + 1
    sols = []
    for d in sp.divisors(target):
        e = target // d
        if d <= e:
            continue
        x = d - 1
        y = e - 1
        if x > 1 and y > 1 and (x != y):
            sols.append((x, y))
    return sols
print(solution_for_s(50))

### Turn 36
solution_for_s(44)


### Turn 36
print(solution_for_s(44))

### Turn 37
M_mod_44 = pow(3, t, 44)
c = 10162574
N_mod_44 = (M_mod_44 + c % 44) % 44
N_mod_44


### Turn 37
M_mod_44 = pow(3, t, 44)
c = 10162574
N_mod_44 = (M_mod_44 + c % 44) % 44
print(N_mod_44)

### Turn 38
def minimal_ratio_by_scanning(N_type, param):
    # return minimal ratio as Fraction by scanning s up to a limit
    # We'll generate s candidates using pair search up to max_s
    best = None
    best_pair = None
    for ratio, x, y, s in candidates:
        # check divisibility
        M_mod_s = M_mod[s]
        if N_type == "offset":
            N_mod_s = (M_mod_s + (param % s)) % s
        else:  # multiple, factor = param (e.g., 5)
            N_mod_s = (M_mod_s * param) % s
        if N_mod_s != 0:
            continue
        # found divisor
        best = ratio
        best_pair = (x, y, s)
        break
    return best, best_pair

# test each N again
for idx, (typ, param) in enumerate(N_info):
    best, pair = minimal_ratio_by_scanning(typ, param)
    print(idx, best, pair)


### Turn 38
def minimal_ratio_by_scanning(N_type, param):
    best = None
    best_pair = None
    for ratio, x, y, s in candidates:
        M_mod_s = M_mod[s]
        if N_type == 'offset':
            N_mod_s = (M_mod_s + param % s) % s
        else:
            N_mod_s = M_mod_s * param % s
        if N_mod_s != 0:
            continue
        best = ratio
        best_pair = (x, y, s)
        break
    return (best, best_pair)
for idx, (typ, param) in enumerate(N_info):
    best, pair = minimal_ratio_by_scanning(typ, param)
    print(idx, best, pair)

### Turn 39
def min_ratio_for_s(s):
    # compute minimal ratio for given s
    import sympy as sp
    target = s+1
    best = None
    for d in sp.divisors(target):
        e = target // d
        if d <= e: continue
        x = d -1
        y = e -1
        if x<=1 or y<=1 or x==y: continue
        ratio=Fraction(x*y, s)
        if best is None or ratio < best:
            best=ratio
    return best

for e in range(2,8):
    s = 3**e
    print(e, s, min_ratio_for_s(s))


### Turn 39
def min_ratio_for_s(s):
    import sympy as sp
    target = s + 1
    best = None
    for d in sp.divisors(target):
        e = target // d
        if d <= e:
            continue
        x = d - 1
        y = e - 1
        if x <= 1 or y <= 1 or x == y:
            continue
        ratio = Fraction(x * y, s)
        if best is None or ratio < best:
            best = ratio
    return best
for e in range(2, 8):
    s = 3 ** e
    print(e, s, min_ratio_for_s(s))

### Turn 40
def compute_best_ratio(N_type, param, max_s=5000):
    best = None
    best_pair = None
    for s in range(12, max_s+1):  # s minimal 12 (since s=11 is for ratio 6/11 which requires s divisible by 11)
        # compute min ratio for this s
        # find x,y>1 distinct solving xy+x+y=s
        # use factorization of s+1
        target = s+1
        # compute divisors of target using sympy (maybe heavy for 5000 loop). We'll compute manually.
        import sympy as sp
        sols = []
        for d in sp.divisors(target):
            e = target // d
            if d <= e:
                continue
            x = d - 1
            y = e - 1
            if x <= 1 or y <= 1 or x == y:
                continue
            ratio = Fraction(x*y, s)
            if best is None or ratio < best:
                # check divisibility of N by s
                M_mod_s = M_mod.get(s)
                if M_mod_s is None:
                    M_mod_s = pow(3, t, s)
                    M_mod[s] = M_mod_s
                if N_type == "offset":
                    N_mod_s = (M_mod_s + (param % s)) % s
                else:
                    N_mod_s = (M_mod_s * param) % s
                if N_mod_s == 0:
                    best = ratio
                    best_pair = (x,y,s)
        # continue scanning
    return best, best_pair

# Compute for each N with max_s maybe 2000
for idx, (typ,param) in enumerate(N_info):
    best, pair = compute_best_ratio(typ,param, max_s=2000)
    print(idx, best, pair)


### Turn 40
def compute_best_ratio(N_type, param, max_s=5000):
    best = None
    best_pair = None
    for s in range(12, max_s + 1):
        target = s + 1
        import sympy as sp
        sols = []
        for d in sp.divisors(target):
            e = target // d
            if d <= e:
                continue
            x = d - 1
            y = e - 1
            if x <= 1 or y <= 1 or x == y:
                continue
            ratio = Fraction(x * y, s)
            if best is None or ratio < best:
                M_mod_s = M_mod.get(s)
                if M_mod_s is None:
                    M_mod_s = pow(3, t, s)
                    M_mod[s] = M_mod_s
                if N_type == 'offset':
                    N_mod_s = (M_mod_s + param % s) % s
                else:
                    N_mod_s = M_mod_s * param % s
                if N_mod_s == 0:
                    best = ratio
                    best_pair = (x, y, s)
    return (best, best_pair)
for idx, (typ, param) in enumerate(N_info):
    best, pair = compute_best_ratio(typ, param, max_s=2000)
    print(idx, best, pair)

### Turn 41
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(34,53) + Fraction(30,47) + Fraction(130,197) + Fraction(110,167)
S, S.numerator, S.denominator


### Turn 41
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(34, 53) + Fraction(30, 47) + Fraction(130, 197) + Fraction(110, 167)
print((S, S.numerator, S.denominator))

