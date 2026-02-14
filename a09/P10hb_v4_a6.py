# Auto-extracted from: P10hb_v4.md
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



### Turn 0
import sympy as sp

def f_bruteforce(n, limit=10000):
    # Find smallest positive integer m that has three distinct positive divisors sum to n
    # Search increasing m.
    for m in range(1, limit+1):
        # find all divisors
        divs = sp.divisors(m)
        # need distinct three positive divisors.
        # Choose any combination of three distinct divisors.
        # Use sum condition.
        # Might use combinations.
        from itertools import combinations
        for a,b,c in combinations(divs, 3):
            if a+b+c == n:
                return m
    return None

def f_formula(n):
    # compute using divisor approach
    # Find largest d dividing n-1 such that 2 <= d <= (n-1)//3 (floor)
    N = n-1
    maxd = None
    for d in sp.divisors(N):
        if d >= 2 and d <= N//3:
            if maxd is None or d > maxd:
                maxd = d
    if maxd is not None:
        return n-1-maxd
    else:
        # fallback? maybe other triple. we need to compute via brute for small n.
        return None

# test for n from 6 to 30
results = []
for n in range(6, 31):
    bf = f_bruteforce(n, limit=2000)
    fm = f_formula(n)
    results.append((n,bf,fm))
results[:20]


### Turn 0
import sympy as sp

def f_bruteforce(n, limit=10000):
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        from itertools import combinations
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                return m
    return None

def f_formula(n):
    N = n - 1
    maxd = None
    for d in sp.divisors(N):
        if d >= 2 and d <= N // 3:
            if maxd is None or d > maxd:
                maxd = d
    if maxd is not None:
        return n - 1 - maxd
    else:
        return None
results = []
for n in range(6, 31):
    bf = f_bruteforce(n, limit=2000)
    fm = f_formula(n)
    results.append((n, bf, fm))
print(results[:20])

### Turn 1
def f_bruteforce_limit(n, limit=500):
    for m in range(1, limit+1):
        divs = sp.divisors(m)
        from itertools import combinations
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                return m
    return None

for n in [14,18,20,24]:
    print(n, f_bruteforce_limit(n, 500))


### Turn 1
def f_bruteforce_limit(n, limit=500):
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        from itertools import combinations
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                return m
    return None
for n in [14, 18, 20, 24]:
    print(n, f_bruteforce_limit(n, 500))

### Turn 2
def triple_for_n(m):
    divs = sp.divisors(m)
    from itertools import combinations
    for a,b,c in combinations(divs,3):
        if a+b+c == m:
            return (a,b,c)
    return None

for m in range(1,31):
    t=triple_for_n(m)
    if t:
        print(m, t)


### Turn 2
def triple_for_n(m):
    divs = sp.divisors(m)
    from itertools import combinations
    for a, b, c in combinations(divs, 3):
        if a + b + c == m:
            return (a, b, c)
    return None
for m in range(1, 31):
    t = triple_for_n(m)
    if t:
        print(m, t)

### Turn 3
def f_bruteforce_fast(n, limit=None):
    # Use bound x >= ceil(2*n/3)
    lower = (2*n + 2)//3  # ceil(2n/3)
    if limit is None:
        limit = n*2  # generous
    for m in range(lower, limit+1):
        # quick check: need three distinct divisors sum to n
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        # Use combinations
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == n:
                return m
    return None

def compute_f_upto(N):
    vals = {}
    for n in range(6,N+1):
        vals[n] = f_bruteforce_fast(n)
    return vals

vals = compute_f_upto(150)
list(vals.items())[:30]


### Turn 3
def f_bruteforce_fast(n, limit=None):
    lower = (2 * n + 2) // 3
    if limit is None:
        limit = n * 2
    for m in range(lower, limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == n:
                return m
    return None

def compute_f_upto(N):
    vals = {}
    for n in range(6, N + 1):
        vals[n] = f_bruteforce_fast(n)
    return vals
vals = compute_f_upto(150)
print(list(vals.items())[:30])

### Turn 4
max_ratio = 0
max_n = None
for n, m in vals.items():
    if m is not None:
        ratio = m/n
        if ratio > max_ratio:
            max_ratio = ratio
            max_n = (n,m)
max_ratio, max_n


### Turn 4
max_ratio = 0
max_n = None
for n, m in vals.items():
    if m is not None:
        ratio = m / n
        if ratio > max_ratio:
            max_ratio = ratio
            max_n = (n, m)
print((max_ratio, max_n))

### Turn 5
def f_bruteforce_limit2(n, maxm=None):
    if maxm is None:
        maxm = int(2*n) + 100
    for m in range(1, maxm+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == n:
                return m
    return None

def compute_f_range(N):
    arr={}
    for n in range(6,N+1):
        arr[n]=f_bruteforce_limit2(n)
    return arr

vals2=compute_f_range(200)
# compare with earlier for first 150
diff=[n for n in range(6,151) if vals2[n]!=vals[n]]
diff[:20]


### Turn 5
def f_bruteforce_limit2(n, maxm=None):
    if maxm is None:
        maxm = int(2 * n) + 100
    for m in range(1, maxm + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == n:
                return m
    return None

def compute_f_range(N):
    arr = {}
    for n in range(6, N + 1):
        arr[n] = f_bruteforce_limit2(n)
    return arr
vals2 = compute_f_range(200)
diff = [n for n in range(6, 151) if vals2[n] != vals[n]]
print(diff[:20])

### Turn 6
def f_brute_exhaustive(n, max_limit=1000):
    # exhaustive up to max_limit
    for m in range(1, max_limit+1):
        divs=sp.divisors(m)
        if len(divs) <3:
            continue
        from itertools import combinations
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                return m
    return None

vals_ex={}
for n in range(6,51):
    vals_ex[n]=f_brute_exhaustive(n, max_limit=2000)
vals_ex


### Turn 6
def f_brute_exhaustive(n, max_limit=1000):
    for m in range(1, max_limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        from itertools import combinations
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                return m
    return None
vals_ex = {}
for n in range(6, 51):
    vals_ex[n] = f_brute_exhaustive(n, max_limit=2000)
print(vals_ex)

### Turn 7
ratios = [(n, vals_ex[n], vals_ex[n]/n) for n in range(6,51)]
ratios[:20]


### Turn 7
ratios = [(n, vals_ex[n], vals_ex[n] / n) for n in range(6, 51)]
print(ratios[:20])

### Turn 8
def floor6(n):
    return (n//6)*6
[(n, vals_ex[n], floor6(n)) for n in range(6,31)]


### Turn 8
def floor6(n):
    return n // 6 * 6
print([(n, vals_ex[n], floor6(n)) for n in range(6, 31)])

### Turn 9
diffs = [(n, n-vals_ex[n]) for n in range(6,51)]
diffs[:20]


### Turn 9
diffs = [(n, n - vals_ex[n]) for n in range(6, 51)]
print(diffs[:20])

### Turn 10
mod6 = {}
for n in range(6,200):
    f = vals_ex[n]
    r = n % 6
    if r not in mod6:
        mod6[r]=set()
    mod6[r].add(f%6)
mod6


### Turn 10
mod6 = {}
for n in range(6, 200):
    f = vals_ex[n]
    r = n % 6
    if r not in mod6:
        mod6[r] = set()
    mod6[r].add(f % 6)
print(mod6)

### Turn 11
def f_brute_limit(n):
    max_m = max(2*n, 500)  # enough
    for m in range(1, max_m+1):
        divs=sp.divisors(m)
        if len(divs) <3:
            continue
        from itertools import combinations
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                return m
    # fallback none
    return None

vals_all={}
for n in range(6,201):
    vals_all[n]=f_brute_limit(n)
len(vals_all)


### Turn 11
def f_brute_limit(n):
    max_m = max(2 * n, 500)
    for m in range(1, max_m + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        from itertools import combinations
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                return m
    return None
vals_all = {}
for n in range(6, 201):
    vals_all[n] = f_brute_limit(n)
print(len(vals_all))

### Turn 12
mod6 = {}
for n,f in vals_all.items():
    r=n%6
    mod6.setdefault(r,set()).add(f%6)
mod6


### Turn 12
mod6 = {}
for n, f in vals_all.items():
    r = n % 6
    mod6.setdefault(r, set()).add(f % 6)
print(mod6)

### Turn 13
vals_mod3 = [(n, vals_all[n]) for n in range(6,201) if n%6==3]
vals_mod3[:20]


### Turn 13
vals_mod3 = [(n, vals_all[n]) for n in range(6, 201) if n % 6 == 3]
print(vals_mod3[:20])

### Turn 14
[(n, vals_all[n]//6) for n in range(6,61) if n%6==3]


### Turn 14
print([(n, vals_all[n] // 6) for n in range(6, 61) if n % 6 == 3])

### Turn 15
def find_f(n):
    return f_brute_limit(n)
find_f(45)


### Turn 15
def find_f(n):
    return f_brute_limit(n)
print(find_f(45))

### Turn 16
find_f(63)


### Turn 16
print(find_f(63))

### Turn 17
def max_divisor_leq(N):
    maxd = None
    for d in sp.divisors(N):
        if d >= 2 and d <= N//3:
            maxd = max(maxd or 0, d)
    return maxd
for n in [9,15,21,27,33,39,45,51,57]:
    N=n-1
    print(n,N,max_divisor_leq(N))


### Turn 17
def max_divisor_leq(N):
    maxd = None
    for d in sp.divisors(N):
        if d >= 2 and d <= N // 3:
            maxd = max(maxd or 0, d)
    return maxd
for n in [9, 15, 21, 27, 33, 39, 45, 51, 57]:
    N = n - 1
    print(n, N, max_divisor_leq(N))

### Turn 18
for n in [9,15,21,27,33,39,45,51,57]:
    N=n-1
    d = max_divisor_leq(N)
    if d is not None:
        m = N - d
    else:
        m = None
    print(n,d,m)


### Turn 18
for n in [9, 15, 21, 27, 33, 39, 45, 51, 57]:
    N = n - 1
    d = max_divisor_leq(N)
    if d is not None:
        m = N - d
    else:
        m = None
    print(n, d, m)

### Turn 19
find_f(42)


### Turn 19
print(find_f(42))

### Turn 20
def triple_sum_to_n(m, n):
    divs = sp.divisors(m)
    for a,b,c in sp.utilities.iterables.combinations(divs,3):
        if a+b+c == n:
            return (a,b,c)
    return None

triple_sum_to_n(18,30)


### Turn 20
def triple_sum_to_n(m, n):
    divs = sp.divisors(m)
    for a, b, c in sp.utilities.iterables.combinations(divs, 3):
        if a + b + c == n:
            return (a, b, c)
    return None
print(triple_sum_to_n(18, 30))

### Turn 21
def smallest_odd_prime_factor(x):
    # returns smallest odd prime divisor >2, or None if none
    for p in sp.primerange(3, 200):
        if x % p == 0:
            return p
    return None

def spf_odd_of_3pow(k):
    val = pow(3, k, 10**12)  # Not enough for huge; we need actual factorization for small k only


### Turn 21
def smallest_odd_prime_factor(x):
    for p in sp.primerange(3, 200):
        if x % p == 0:
            return p
    return None

def spf_odd_of_3pow(k):
    val = pow(3, k, 10 ** 12)

### Turn 22
import sympy as sp
sp.factorint(1848374)


### Turn 22
import sympy as sp
print(sp.factorint(1848374))

### Turn 23
1848374 % 3


### Turn 23
print(1848374 % 3)

### Turn 24
c3 = 10162574
c3 % 3


### Turn 24
c3 = 10162574
print(c3 % 3)

### Turn 25
sp.factorint(c3)


### Turn 25
print(sp.factorint(c3))

### Turn 26
sp.isprime(5081287)


### Turn 26
print(sp.isprime(5081287))

### Turn 27
c4 = 265710644
c4 % 3


### Turn 27
c4 = 265710644
print(c4 % 3)

### Turn 28
sp.factorint(c4)


### Turn 28
print(sp.factorint(c4))

### Turn 29
c5 = 44636594
c5 % 3


### Turn 29
c5 = 44636594
print(c5 % 3)

### Turn 30
sp.factorint(c5)


### Turn 30
print(sp.factorint(c5))

### Turn 31
from fractions import Fraction
import math
# define symbolic with denominator 2025! maybe huge, but we can treat as Fraction.
# Let's compute 2025! (as integer) using sympy
from sympy import factorial
F = factorial(2025)
F_int = int(F)  # huge but maybe memory heavy; but we might not need actual value.


### Turn 31
from fractions import Fraction
import math
from sympy import factorial
F = factorial(2025)
F_int = int(F)
print(F_int)

### Turn 32
from fractions import Fraction
S0 = Fraction(4,5) + Fraction(5,1) + Fraction(1,1) + Fraction(10,11) + Fraction(28,29) + Fraction(16,17)
S0


### Turn 32
from fractions import Fraction
S0 = Fraction(4, 5) + Fraction(5, 1) + Fraction(1, 1) + Fraction(10, 11) + Fraction(28, 29) + Fraction(16, 17)
print(S0)

### Turn 33
S0.numerator, S0.denominator


### Turn 33
print((S0.numerator, S0.denominator))

### Turn 34
q = 260732 // 27115
q


### Turn 34
q = 260732 // 27115
print(q)

### Turn 35
260732 - 9*27115


### Turn 35
print(260732 - 9 * 27115)

### Turn 36
import sympy as sp
L = sp.factorial(2025)
p = 260732*L - 81345
q = 27115*L
sp.gcd(p,q)


### Turn 36
import sympy as sp
L = sp.factorial(2025)
p = 260732 * L - 81345
q = 27115 * L
print(sp.gcd(p, q))

### Turn 37
g = 81345
p_red = sp.simplify(p // g)
q_red = sp.simplify(q // g)
p_red_mod = sp.mod(p_red, 99991)
q_red_mod = sp.mod(q_red, 99991)
p_red_mod, q_red_mod


### Turn 37
g = 81345
p_red = sp.simplify(p // g)
q_red = sp.simplify(q // g)
p_red_mod = sp.mod(p_red, 99991)
q_red_mod = sp.mod(q_red, 99991)
print((p_red_mod, q_red_mod))

### Turn 38
p_int = int(p // g)
q_int = int(q // g)
len(str(p_int)), len(str(q_int))


### Turn 38
p_int = int(p // g)
q_int = int(q // g)
print((len(str(p_int)), len(str(q_int))))

### Turn 39
import math, sympy as sp
math.gcd(81345, 99991)


### Turn 39
import math, sympy as sp
print(math.gcd(81345, 99991))

### Turn 40
sp.isprime(99991)


### Turn 40
print(sp.isprime(99991))

### Turn 41
mod = 99991
L_mod = 1
for i in range(1,2026):
    L_mod = (L_mod * i) % mod
L_mod


### Turn 41
mod = 99991
L_mod = 1
for i in range(1, 2026):
    L_mod = L_mod * i % mod
print(L_mod)

### Turn 42
287847 % mod


### Turn 42
print(287847 % mod)

### Turn 43
num = (87865 * L_mod - 81345) % mod
num


### Turn 43
num = (87865 * L_mod - 81345) % mod
print(num)

### Turn 44
inv = sp.mod_inverse(81345, mod)
inv


### Turn 44
inv = sp.mod_inverse(81345, mod)
print(inv)

### Turn 45
result = (num * inv) % mod
result


### Turn 45
result = num * inv % mod
print(result)

### Turn 46
# compute p_red_mod directly
inv = sp.mod_inverse(81345, mod)
p_red_mod = ((260732 * L_mod - 81345) % mod) * inv % mod
q_red_mod = ((27115 * L_mod) % mod) * inv % mod
(p_red_mod + q_red_mod) % mod


### Turn 46
inv = sp.mod_inverse(81345, mod)
p_red_mod = (260732 * L_mod - 81345) % mod * inv % mod
q_red_mod = 27115 * L_mod % mod * inv % mod
print((p_red_mod + q_red_mod) % mod)

### Turn 47
def f_bruteforce_general(N, max_search=5000):
    # compute f(N) using earlier brute search up to maybe 2N
    limit = max(2*N, 5000)
    for m in range(1, limit+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == N:
                return m
    return None

def g_bruteforce(M, c, F):
    N = M + c
    fN = f_bruteforce_general(N, max_search=20000)
    k = (F * fN) // M
    return Fraction(k, F)

def predict_g(M, c, F):
    N = M + c
    # Determine p as smallest odd prime divisor of N-1
    Nminus1 = N - 1
    # find smallest odd prime divisor
    def smallest_odd_prime_factor(x):
        # return smallest odd prime divisor >2
        for p in sp.primerange(3, 1000):
            if x % p == 0:
                return p
        # if none found up to limit, assume x is prime > 1000
        return None
    # Need to handle special case where N-1 has no odd divisor <= limit
    p = smallest_odd_prime_factor(Nminus1)
    # If not found, treat as large p
    if p is None:
        # treat as p large > F (like >2025)
        p_large = True
        p_val = None
    else:
        p_large = p > F  # analogous to p > 2025
        p_val = p
    # Determine t = floor(c/M)
    t = c // M
    # Determine if c == 0 (special negative epsilon)
    # Use derived formulas
    if c == 0:
        # use p = smallest odd prime divisor of M-1 (should be 5)
        p = p_val
        # g = (p-1)/p - 1/F
        return Fraction(p-1, p) - Fraction(1, F)
    else:
        if p_large:
            # p does not divide F
            # g = t + ( (F-1)/F )? Wait formula for large p:
            # g = t + (1 - 1/F)?? Actually for c<M and p large, g = 1 - 1/F
            # For t=0: g = 1 - 1/F
            # For t>0: g = (t+1) - 1/F ? Let's compute
            # Derived earlier: if α not integer and t integer >=0, then k = floor(α*(t+1) - α/M) = floor((t+1)*α - ε) = floor((t+1)*α) - 1 because α is just less than integer
            # α = F - δ (δ small). So (t+1)*α = (t+1)*F - (t+1)*δ. Since (t+1)*δ <1? For large p, δ=F/p<1, (t+1)*δ may be <1 if p > (t+1)*F. For our case where p huge >F, and t small (0 or 4), (t+1)*δ < (t+1)*F/p < (t+1)*F/F = t+1 < maybe? Actually p>F => δ<1, (t+1)*δ < t+1, may be >=1 if t+1 >1. For t=4, (t+1)*δ <5. Could be >1. This leads to floor((t+1)*α) = (t+1)*F - ceil((t+1)*δ). Not trivial.
            pass


### Turn 47
def f_bruteforce_general(N, max_search=5000):
    limit = max(2 * N, 5000)
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == N:
                return m
    return None

def g_bruteforce(M, c, F):
    N = M + c
    fN = f_bruteforce_general(N, max_search=20000)
    k = F * fN // M
    return Fraction(k, F)

def predict_g(M, c, F):
    N = M + c
    Nminus1 = N - 1

    def smallest_odd_prime_factor(x):
        for p in sp.primerange(3, 1000):
            if x % p == 0:
                return p
        return None
    p = smallest_odd_prime_factor(Nminus1)
    if p is None:
        p_large = True
        p_val = None
    else:
        p_large = p > F
        p_val = p
    t = c // M
    if c == 0:
        p = p_val
        return Fraction(p - 1, p) - Fraction(1, F)
    elif p_large:
        pass

### Turn 48
def compute_f(N):
    # Use heuristic formula: try 1,d,m representation if possible, else use multiple approach search up to maybe 2N.
    # We'll use brute for reliability for small N (since N moderate)
    for m in range(1, 5*N+10):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == N:
                return m
    return None

def g_bruteforce(M, c, F):
    N = M + c
    fN = compute_f(N)
    k = (F * fN) // M
    return Fraction(k, F)

def test_cases():
    K=6
    M = 3**K
    F = math.factorial(K)  # using K! for test
    cases = [0, 4*M, 100, 200, 500, M-1]
    for c in cases:
        g = g_bruteforce(M,c,F)
        print("c=",c,"g=",g)
test_cases()


### Turn 48
def compute_f(N):
    for m in range(1, 5 * N + 10):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == N:
                return m
    return None

def g_bruteforce(M, c, F):
    N = M + c
    fN = compute_f(N)
    k = F * fN // M
    return Fraction(k, F)

def test_cases():
    K = 6
    M = 3 ** K
    F = math.factorial(K)
    cases = [0, 4 * M, 100, 200, 500, M - 1]
    for c in cases:
        g = g_bruteforce(M, c, F)
        print('c=', c, 'g=', g)
print(test_cases())

### Turn 49
f_729 = compute_f(729)
f_729


### Turn 49
f_729 = compute_f(729)
print(f_729)

### Turn 50
def triple_for_n_m(n):
    for m in range(1,2000):
        divs = sp.divisors(m)
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == n:
                return (m,a,b,c)
    return None
triple_for_n_m(729)


### Turn 50
def triple_for_n_m(n):
    for m in range(1, 2000):
        divs = sp.divisors(m)
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == n:
                return (m, a, b, c)
    return None
print(triple_for_n_m(729))

### Turn 51
cs = [0, 4* (3**2025), 1848374, 10162574, 265710644, 44636594]
# compute small c mod 9
mods = [c % 9 for c in [0, 1848374, 10162574, 265710644, 44636594]]
mods


### Turn 51
cs = [0, 4 * 3 ** 2025, 1848374, 10162574, 265710644, 44636594]
mods = [c % 9 for c in [0, 1848374, 10162574, 265710644, 44636594]]
print(mods)

### Turn 52
[1848374%9, 10162574%9, 265710644%9, 44636594%9]


### Turn 52
print([1848374 % 9, 10162574 % 9, 265710644 % 9, 44636594 % 9])

### Turn 53
265710644 % 9


### Turn 53
print(265710644 % 9)

### Turn 54
10162574 % 9


### Turn 54
print(10162574 % 9)

### Turn 55
44636594 % 9


### Turn 55
print(44636594 % 9)

### Turn 56
[1848374%3, 10162574%3, 265710644%3, 44636594%3]


### Turn 56
print([1848374 % 3, 10162574 % 3, 265710644 % 3, 44636594 % 3])

### Turn 57
[1848374%5, 10162574%5, 265710644%5, 44636594%5]


### Turn 57
print([1848374 % 5, 10162574 % 5, 265710644 % 5, 44636594 % 5])

### Turn 58
[1848374%7, 10162574%7, 265710644%7, 44636594%7]


### Turn 58
print([1848374 % 7, 10162574 % 7, 265710644 % 7, 44636594 % 7])

### Turn 59
from fractions import Fraction
sum_frac = Fraction(10,11) + Fraction(28,29) + Fraction(16,17)
sum_frac


### Turn 59
from fractions import Fraction
sum_frac = Fraction(10, 11) + Fraction(28, 29) + Fraction(16, 17)
print(sum_frac)

### Turn 60
sp.isprime(5423)


### Turn 60
print(sp.isprime(5423))

### Turn 61
sp.factorint(5423)


### Turn 61
print(sp.factorint(5423))

### Turn 62
mod = 99991
F_mod = 57239
inv_5423 = sp.mod_inverse(5423, mod)
value = (F_mod * 47808 * inv_5423) % mod
result = (value - 1) % mod
result


### Turn 62
mod = 99991
F_mod = 57239
inv_5423 = sp.mod_inverse(5423, mod)
value = F_mod * 47808 * inv_5423 % mod
result = (value - 1) % mod
print(result)

### Turn 63
F_mod2 = sp.factorial(2025) % mod
F_mod2


### Turn 63
F_mod2 = sp.factorial(2025) % mod
print(F_mod2)

### Turn 64
from fractions import Fraction
F = Fraction(5,1) + Fraction(10,11) + Fraction(28,29) + Fraction(16,17) - Fraction(1, 2025*sp.factorial(2024)) # this is 1/F? Wait 2025! is huge, not compute.


### Turn 64
from fractions import Fraction
F = Fraction(5, 1) + Fraction(10, 11) + Fraction(28, 29) + Fraction(16, 17) - Fraction(1, 2025 * sp.factorial(2024))
print(F)

### Turn 65
import sys, math
sys.set_int_max_str_digits(10000)
F_int = math.factorial(2025)
# compute sum S rational
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(10,11) + Fraction(1,1) - Fraction(1, F_int) + Fraction(28,29) + Fraction(16,17)
# Wait we must incorporate g(0) = 2/3, g(4M)=10/3, etc.
# S = 2/3 + 10/3 + 10/11 + (1 - 1/F) + 28/29 + 16/17
S = Fraction(2,3) + Fraction(10,3) + Fraction(10,11) + Fraction(1,1) - Fraction(1, F_int) + Fraction(28,29) + Fraction(16,17)
S_numer = S.numerator
S_denom = S.denominator
# Reduce fraction (should be already reduced)
from math import gcd
g = math.gcd(S_numer, S_denom)
g


### Turn 65
import sys, math
sys.set_int_max_str_digits(10000)
F_int = math.factorial(2025)
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(10, 11) + Fraction(1, 1) - Fraction(1, F_int) + Fraction(28, 29) + Fraction(16, 17)
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(10, 11) + Fraction(1, 1) - Fraction(1, F_int) + Fraction(28, 29) + Fraction(16, 17)
S_numer = S.numerator
S_denom = S.denominator
from math import gcd
g = math.gcd(S_numer, S_denom)
print(g)

### Turn 66
mod = 99991
rem = (S_numer + S_denom) % mod
rem


### Turn 66
mod = 99991
rem = (S_numer + S_denom) % mod
print(rem)

### Turn 67
def f_bruteforce(N):
    # brute up to 3N maybe
    for m in range(1, 3*N+10):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in sp.utilities.iterables.combinations(divs,3):
            if a+b+c == N:
                return m
    return None

f81 = f_bruteforce(81)
f81


### Turn 67
def f_bruteforce(N):
    for m in range(1, 3 * N + 10):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in sp.utilities.iterables.combinations(divs, 3):
            if a + b + c == N:
                return m
    return None
f81 = f_bruteforce(81)
print(f81)

