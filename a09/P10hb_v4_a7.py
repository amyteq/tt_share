# Auto-extracted from: P10hb_v4.md
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



### Turn 0
c1 = 1848374
c2 = 10162574
c3 = 265710644
c4 = 44636594
[c%4 for c in [c1,c2,c3,c4]]


### Turn 0
c1 = 1848374
c2 = 10162574
c3 = 265710644
c4 = 44636594
print([c % 4 for c in [c1, c2, c3, c4]])

### Turn 1
cs = [0, 4, 1848374, 10162574, 265710644, 44636594]
[(c-1)%3 for c in cs]


### Turn 1
cs = [0, 4, 1848374, 10162574, 265710644, 44636594]
print([(c - 1) % 3 for c in cs])

### Turn 2
import sympy as sp
c_vals = [1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c))


### Turn 2
import sympy as sp
c_vals = [1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    print(c, sp.factorint(c))

### Turn 3
import math, sympy as sp, sys, itertools, functools, fractions, math, random, decimal, os, time, hashlib, json, collections, itertools, fractions, typing, sys, math
A = math.factorial(2025)
A_bitlen = A.bit_length()
A_bitlen


### Turn 3
import math, sympy as sp, sys, itertools, functools, fractions, math, random, decimal, os, time, hashlib, json, collections, itertools, fractions, typing, sys, math
A = math.factorial(2025)
A_bitlen = A.bit_length()
print(A_bitlen)

### Turn 4
d_vals = [4,4,11,5081287,4,17]
ceil_sum = sum( (A + d - 1)//d for d in d_vals )  # ceil division
ceil_sum


### Turn 4
d_vals = [4, 4, 11, 5081287, 4, 17]
ceil_sum = sum(((A + d - 1) // d for d in d_vals))
print(ceil_sum)

### Turn 5
import math, sys
sys.set_int_max_str_digits(10000)  # increase limit
# compute ceil sum
ceil_sum = sum((A + d - 1)//d for d in d_vals)
numer = 7*A - ceil_sum
denom = A
g = math.gcd(numer, denom)
p = numer // g
q = denom // g
(p+q) % 99991


### Turn 5
import math, sys
sys.set_int_max_str_digits(10000)
ceil_sum = sum(((A + d - 1) // d for d in d_vals))
numer = 7 * A - ceil_sum
denom = A
g = math.gcd(numer, denom)
p = numer // g
q = denom // g
print((p + q) % 99991)

### Turn 6
def f_brutal(n, limit=10000):
    # find smallest positive integer N with three distinct divisors summing to n
    N = 1
    while True:
        # compute divisors
        divisors = [d for d in range(1, int(N**0.5)+1) if N % d == 0]
        all_divs = set()
        for d in divisors:
            all_divs.add(d)
            all_divs.add(N//d)
        # distinct list
        divs = sorted(all_divs)
        # need three distinct divisors with sum n
        # we can just try combinations of 3 from divs
        L = len(divs)
        for i in range(L):
            for j in range(i+1, L):
                for k in range(j+1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
        N += 1
        if N > limit:
            return None

def f_formula(n):
    s = n-1
    # find smallest divisor >2 of s
    # search from 3 upward
    for d in range(3, int(s**0.5)+1):
        if s % d == 0:
            # if d == s//d? but d <= sqrt(s). Smallest divisor >2 can be d or maybe s//d could be smaller >2.
            # However we need the smallest divisor greater than 2 (any divisor). So we should also consider complement if it's smaller?
            pass
    # Simpler: compute all divisors >2 and take min
    divs = [d for d in range(3, s+1) if s % d == 0]
    if not divs:
        # s prime
        # smallest divisor >2 not exist
        # treat as case odd prime -> use a=2? But formula earlier: f = 2s -4.
        # We'll compute accordingly.
        return 2*s - 4
    # find smallest divisor >2
    d_min = min(divs)
    # compute f = s - s/d_min
    return s - s//d_min

# test for n up to 200
bad = []
for n in range(6, 200):
    f1 = f_brutal(n, limit=5000)
    f2 = f_formula(n)
    if f1 != f2:
        bad.append((n,f1,f2))
bad[:10], len(bad)


### Turn 6
def f_brutal(n, limit=10000):
    N = 1
    while True:
        divisors = [d for d in range(1, int(N ** 0.5) + 1) if N % d == 0]
        all_divs = set()
        for d in divisors:
            all_divs.add(d)
            all_divs.add(N // d)
        divs = sorted(all_divs)
        L = len(divs)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
        N += 1
        if N > limit:
            return None

def f_formula(n):
    s = n - 1
    for d in range(3, int(s ** 0.5) + 1):
        if s % d == 0:
            pass
    divs = [d for d in range(3, s + 1) if s % d == 0]
    if not divs:
        return 2 * s - 4
    d_min = min(divs)
    return s - s // d_min
bad = []
for n in range(6, 200):
    f1 = f_brutal(n, limit=5000)
    f2 = f_formula(n)
    if f1 != f2:
        bad.append((n, f1, f2))
print((bad[:10], len(bad)))

### Turn 7
def f_formula_debug(n):
    s = n-1
    # find smallest divisor >2
    d_min = None
    for d in range(3, s+1):
        if s % d == 0:
            d_min = d
            break
    if d_min is None:
        # prime case
        return 2*s - 4
    return s - s//d_min

for n in range(6, 16):
    print(n, f_brutal(n, limit=200), f_formula_debug(n))


### Turn 7
def f_formula_debug(n):
    s = n - 1
    d_min = None
    for d in range(3, s + 1):
        if s % d == 0:
            d_min = d
            break
    if d_min is None:
        return 2 * s - 4
    return s - s // d_min
for n in range(6, 16):
    print(n, f_brutal(n, limit=200), f_formula_debug(n))

### Turn 8
def f_formula2(n):
    s = n-1
    # candidate from 2 and s-2
    N2 = 2*(s-2)
    # find smallest divisor >2 of s, if any
    d_min = None
    for d in range(3, int(s**0.5)+1):
        if s % d == 0:
            # we need smallest divisor >2, but the divisor could be larger than sqrt(s) as complement.
            # The smallest divisor >2 is actually min(d, s//d) if both >2.
            # Since d <= sqrt(s), its complement may be bigger; minimal divisor >2 is d (if d>2)
            d_min = d
            break
    if d_min is None:
        # s is prime or s = 2 * prime? Wait s might be a power of 2? If s=4, smallest divisor >2 is 4 (since divisors:4)
        # For s even and not divisible by 4, we may have smallest divisor >2 as maybe something else.
        # Let's handle the case s even:
        if s % 2 == 0:
            # s even; we already excluded divisor 2 (since that leads to a=b)
            # need to find smallest divisor >2; could be 4 if s %4==0 else maybe other.
            # We can just check from 4 upwards
            for d in range(4, int(s**0.5)+1):
                if s % d == 0:
                    d_min = d
                    break
    # Actually we need to consider all divisors >2, not only up to sqrt(s). The smallest >2 divisor might be > sqrt(s) only if s is prime, but then no divisor.
    # So if s is composite, there will be a divisor <= sqrt(s). So above loop suffices.
    if d_min is None:
        # No divisor >2 found (s prime) 
        N1 = None
    else:
        N1 = s - s//d_min
    if N1 is None:
        return N2
    else:
        return min(N1, N2)

bad = []
for n in range(6, 200):
    f1 = f_brutal(n, limit=5000)
    f2 = f_formula2(n)
    if f1 != f2:
        bad.append((n,f1,f2))
len(bad), bad[:10]


### Turn 8
def f_formula2(n):
    s = n - 1
    N2 = 2 * (s - 2)
    d_min = None
    for d in range(3, int(s ** 0.5) + 1):
        if s % d == 0:
            d_min = d
            break
    if d_min is None:
        if s % 2 == 0:
            for d in range(4, int(s ** 0.5) + 1):
                if s % d == 0:
                    d_min = d
                    break
    if d_min is None:
        N1 = None
    else:
        N1 = s - s // d_min
    if N1 is None:
        return N2
    else:
        return min(N1, N2)
bad = []
for n in range(6, 200):
    f1 = f_brutal(n, limit=5000)
    f2 = f_formula2(n)
    if f1 != f2:
        bad.append((n, f1, f2))
print((len(bad), bad[:10]))

### Turn 9
def min_lcm_sum(s):
    best = None
    best_pair = None
    for a in range(2, s//2 + 1):
        b = s - a
        if b <= a:
            continue
        # compute lcm
        l = a*b // math.gcd(a,b)
        if best is None or l < best:
            best = l
            best_pair = (a,b)
    return best, best_pair

def f_from_s(s):
    L,_ = min_lcm_sum(s)
    return L

# test for s up to 30
vals = []
for s in range(5,31):
    f = f_from_s(s)
    vals.append((s,f))
vals[:10]


### Turn 9
def min_lcm_sum(s):
    best = None
    best_pair = None
    for a in range(2, s // 2 + 1):
        b = s - a
        if b <= a:
            continue
        l = a * b // math.gcd(a, b)
        if best is None or l < best:
            best = l
            best_pair = (a, b)
    return (best, best_pair)

def f_from_s(s):
    L, _ = min_lcm_sum(s)
    return L
vals = []
for s in range(5, 31):
    f = f_from_s(s)
    vals.append((s, f))
print(vals[:10])

### Turn 10
def smallest_divisor_gt2(s):
    for d in range(3, int(s**0.5)+1):
        if s % d == 0:
            return d
    # if s even maybe 4?
    if s % 2 == 0:
        if s % 4 == 0:
            return 4
        # else need to check larger composite divisors
        # check from 5 upward
        for d in range(5, int(s**0.5)+1):
            if s % d == 0:
                return d
    # if not found, return None (prime)
    return None

def compare(limit=30):
    mismatches = []
    for s in range(5, limit):
        L, pair = min_lcm_sum(s)
        d = smallest_divisor_gt2(s)
        if d is None:
            # prime case use N2 = 2*(s-2)
            L1 = 2*(s-2)
        else:
            L1 = s - s//d
        if L != L1:
            mismatches.append((s,L,pair,d,L1))
    return mismatches

m = compare(100)
len(m), m[:10]


### Turn 10
def smallest_divisor_gt2(s):
    for d in range(3, int(s ** 0.5) + 1):
        if s % d == 0:
            return d
    if s % 2 == 0:
        if s % 4 == 0:
            return 4
        for d in range(5, int(s ** 0.5) + 1):
            if s % d == 0:
                return d
    return None

def compare(limit=30):
    mismatches = []
    for s in range(5, limit):
        L, pair = min_lcm_sum(s)
        d = smallest_divisor_gt2(s)
        if d is None:
            L1 = 2 * (s - 2)
        else:
            L1 = s - s // d
        if L != L1:
            mismatches.append((s, L, pair, d, L1))
    return mismatches
m = compare(100)
print((len(m), m[:10]))

### Turn 11
print(smallest_divisor_gt2(6))


### Turn 11
print(smallest_divisor_gt2(6))

### Turn 12
print(f_brutal(11, limit=200))


### Turn 12
print(f_brutal(11, limit=200))

### Turn 13
def min_N_over_n(limitN=500):
    ratios = {}
    for N in range(2, limitN+1):
        # get all distinct divisor triples
        divs = set()
        for d in range(1, int(N**0.5)+1):
            if N % d == 0:
                divs.add(d)
                divs.add(N//d)
        divs = sorted(divs)
        L = len(divs)
        for i in range(L):
            for j in range(i+1, L):
                for k in range(j+1, L):
                    s = divs[i] + divs[j] + divs[k]
                    # update minimal N for this sum
                    if s not in ratios or N < ratios[s][0]:
                        ratios[s] = (N, (divs[i], divs[j], divs[k]))
    # compute minimal ratio N/n for all n
    min_ratio = 1
    best = None
    for n, (N, triple) in ratios.items():
        r = N / n
        if r < min_ratio:
            min_ratio = r
            best = (n,N,triple,r)
    return min_ratio, best, ratios

min_ratio, best, ratios = min_N_over_n(500)
min_ratio, best


### Turn 13
def min_N_over_n(limitN=500):
    ratios = {}
    for N in range(2, limitN + 1):
        divs = set()
        for d in range(1, int(N ** 0.5) + 1):
            if N % d == 0:
                divs.add(d)
                divs.add(N // d)
        divs = sorted(divs)
        L = len(divs)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    s = divs[i] + divs[j] + divs[k]
                    if s not in ratios or N < ratios[s][0]:
                        ratios[s] = (N, (divs[i], divs[j], divs[k]))
    min_ratio = 1
    best = None
    for n, (N, triple) in ratios.items():
        r = N / n
        if r < min_ratio:
            min_ratio = r
            best = (n, N, triple, r)
    return (min_ratio, best, ratios)
min_ratio, best, ratios = min_N_over_n(500)
print((min_ratio, best))

### Turn 14
cs = [0, 4, 1848374, 10162574, 265710644, 44636594]
mods = [(1 + c) % 11 for c in cs]  # Since M ≡1
mods


### Turn 14
cs = [0, 4, 1848374, 10162574, 265710644, 44636594]
mods = [(1 + c) % 11 for c in cs]
print(mods)

### Turn 15
pairs = []
for a in range(2, 15):
    for b in range(a+1, 15):
        ab = a*b
        D = ab + a + b
        g = math.gcd(ab, D)
        Dp = D // g
        r = (ab // g) / Dp  # simplified ratio numerator/denominator
        # store rational as tuple
        pairs.append((a,b, Dp, Fraction = (ab//g, Dp), r))
sorted(pairs, key=lambda x: x[3][0]/x[3][1])[:10]


### Turn 15
pairs = []
for a in range(2, 15):
    for b in range(a+1, 15):
        ab = a*b
        D = ab + a + b
        g = math.gcd(ab, D)
        Dp = D // g
        r = (ab // g) / Dp  # simplified ratio numerator/denominator
        # store rational as tuple
        pairs.append((a,b, Dp, Fraction = (ab//g, Dp), r))
sorted(pairs, key=lambda x: x[3][0]/x[3][1])[:10]


### Turn 16
from fractions import Fraction
pairs = []
for a in range(2, 20):
    for b in range(a+1, 20):
        ab = a*b
        D = ab + a + b
        g = math.gcd(ab, D)
        num = ab // g
        den = D // g
        r = Fraction(num, den)
        pairs.append((a,b, den, r))
pairs_sorted = sorted(pairs, key=lambda x: x[3])
pairs_sorted[:10]


### Turn 16
from fractions import Fraction
pairs = []
for a in range(2, 20):
    for b in range(a + 1, 20):
        ab = a * b
        D = ab + a + b
        g = math.gcd(ab, D)
        num = ab // g
        den = D // g
        r = Fraction(num, den)
        pairs.append((a, b, den, r))
pairs_sorted = sorted(pairs, key=lambda x: x[3])
print(pairs_sorted[:10])

### Turn 17
import math, sympy as sp, itertools, sys, math, functools, collections, random, time, os, sys, math
A = math.factorial(2025)  # huge but Python can handle
len(str(A))


### Turn 17
import math, sympy as sp, itertools, sys, math, functools, collections, random, time, os, sys, math
A = math.factorial(2025)
print(len(str(A)))

### Turn 18
from math import gcd
import math, itertools, sys, sympy as sp

A = math.factorial(2025)
M = pow(3, A)  # this will be monstrous, cannot compute directly as integer (huge). But pow with huge exponent will produce huge number; not feasible.


### Turn 18
from math import gcd
import math, itertools, sys, sympy as sp
A = math.factorial(2025)
M = pow(3, A)
print(M)

### Turn 19
import math, time, sys
A = math.factorial(2025)
def M_mod(mod):
    return pow(3, A, mod)

start = time.time()
m11 = M_mod(11)
elapsed = time.time() - start
print('M mod 11 =', m11, 'time', elapsed)


### Turn 19
import math, time, sys
A = math.factorial(2025)

def M_mod(mod):
    return pow(3, A, mod)
start = time.time()
m11 = M_mod(11)
elapsed = time.time() - start
print('M mod 11 =', m11, 'time', elapsed)

### Turn 20
def candidate_pairs(limit_a=50):
    candidates = []
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab = a*b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            ratio = Fraction(num, den)
            candidates.append((ratio, den, num, a, b))
    # sort by ratio ascending (smaller ratio means smaller N/n)
    candidates.sort(key=lambda x: x[0])
    return candidates

cands = candidate_pairs(30)
cands[:10]


### Turn 20
def candidate_pairs(limit_a=50):
    candidates = []
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            ratio = Fraction(num, den)
            candidates.append((ratio, den, num, a, b))
    candidates.sort(key=lambda x: x[0])
    return candidates
cands = candidate_pairs(30)
print(cands[:10])

### Turn 21
from fractions import Fraction

c_values = [0, None, 1848374, 10162574, 265710644, 44636594]  # placeholder for 4M
# We'll treat 4M separately

def compute_ratio_for_n(c, is_fourM=False):
    # c is either integer or placeholder for 4M
    # return minimal ratio Fraction
    # Use candidates up to limit
    for ratio, den, num, a, b in cands:
        # compute n mod den
        M_mod = pow(3, A, den)
        if is_fourM:
            n_mod = (M_mod + (4 * M_mod) ) % den  # n = M + 4M = 5M, but careful: n = M + c where c=4M
            # Actually n = M + c, with c = 4M => n = 5M
            n_mod = (5 * M_mod) % den
        else:
            n_mod = (M_mod + c) % den
        if n_mod == 0:
            return ratio, den, num, a, b
    # fallback to divisor method
    # compute s = n-1 = M + c -1
    # need smallest divisor >2 of s
    # find if s divisible by 4
    M_mod_4 = pow(3, A, 4)
    # compute s_mod_4
    if is_fourM:
        # c = 4M, s = M + 4M -1 = 5M -1
        # compute 5M mod 4 => (5 * M_mod_4) %4; then subtract 1
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        # ratio = (d-1)/d = 3/4
        return Fraction(3,4), d, d-1, None, None
    # else need smallest odd divisor >2 of s.
    # search odd primes up to some limit
    # compute s modulo p
    for p in sp.primerange(3, 5000):
        # compute s mod p
        M_mod_p = pow(3, A, p)
        if is_fourM:
            # s = 5M -1
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            return Fraction(d-1, d), d, d-1, None, None
    # if no small divisor found, treat s as prime => use a=2?
    # fallback: ratio using a=2,b=s-2? That yields ratio approx 2(s-2)/ (s+1) ~ 2.
    # but this is huge, not minimal. 
    # For safety return a huge ratio
    return Fraction(1,1), None, None, None, None

# compute ratios for each c
ratios = []
for idx, c in enumerate(cs):
    if idx == 1:  # 4M case
        ratio_info = compute_ratio_for_n(c, is_fourM=True)
    else:
        ratio_info = compute_ratio_for_n(c, is_fourM=False)
    ratios.append(ratio_info)
ratios


### Turn 21
from fractions import Fraction
c_values = [0, None, 1848374, 10162574, 265710644, 44636594]

def compute_ratio_for_n(c, is_fourM=False):
    for ratio, den, num, a, b in cands:
        M_mod = pow(3, A, den)
        if is_fourM:
            n_mod = (M_mod + 4 * M_mod) % den
            n_mod = 5 * M_mod % den
        else:
            n_mod = (M_mod + c) % den
        if n_mod == 0:
            return (ratio, den, num, a, b)
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        return (Fraction(3, 4), d, d - 1, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            return (Fraction(d - 1, d), d, d - 1, None, None)
    return (Fraction(1, 1), None, None, None, None)
ratios = []
for idx, c in enumerate(cs):
    if idx == 1:
        ratio_info = compute_ratio_for_n(c, is_fourM=True)
    else:
        ratio_info = compute_ratio_for_n(c, is_fourM=False)
    ratios.append(ratio_info)
print(ratios)

### Turn 22
# find any candidate with ratio 2/3
for r, den, num, a,b in cands:
    if r == Fraction(2,3):
        print('found', a,b, den, num)


### Turn 22
for r, den, num, a, b in cands:
    if r == Fraction(2, 3):
        print('found', a, b, den, num)

### Turn 23
def n_mod_den(c, den, is_fourM=False):
    M_mod = pow(3, A, den)
    if is_fourM:
        n_mod = (5 * M_mod) % den  # n = 5M
    else:
        n_mod = (M_mod + c) % den
    return n_mod

def test_mod():
    den = 3
    print("M mod 3:", pow(3, A, 3))
    for c in cs:
        print(c, n_mod_den(c, 3))
    # test for 4M case
    print("4M case mod3:", n_mod_den(0, 3, is_fourM=True))
    
test_mod()


### Turn 23
def n_mod_den(c, den, is_fourM=False):
    M_mod = pow(3, A, den)
    if is_fourM:
        n_mod = 5 * M_mod % den
    else:
        n_mod = (M_mod + c) % den
    return n_mod

def test_mod():
    den = 3
    print('M mod 3:', pow(3, A, 3))
    for c in cs:
        print(c, n_mod_den(c, 3))
    print('4M case mod3:', n_mod_den(0, 3, is_fourM=True))
print(test_mod())

### Turn 24
def minimal_ratio_for_n(c, is_fourM=False):
    for ratio, den, num, a,b in cands:
        if n_mod_den(c, den, is_fourM) == 0:
            # ratio = num/den
            return ratio, den, num, a, b
    # If none found, fallback to divisor method (d)
    # compute s = n-1 = M + c - 1
    # compute smallest divisor >2 of s: check if s %4 ==0 else smallest odd prime factor
    # Use modular checks
    # First compute s_mod_4
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        # n = 5M, s = 5M -1
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        ratio = Fraction(d-1, d)  # 3/4
        return ratio, d, d-1, None, None
    # else find smallest odd prime divisor
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            ratio = Fraction(d-1, d)
            return ratio, d, d-1, None, None
    # fallback large prime (should not happen)
    # use ratio near 1 maybe
    return Fraction(1,1), None, None, None, None

# compute for each c
results = []
for idx,c in enumerate(cs):
    if idx == 1: # 4M case
        res = minimal_ratio_for_n(c, is_fourM=True)
    else:
        res = minimal_ratio_for_n(c, is_fourM=False)
    results.append(res)
results


### Turn 24
def minimal_ratio_for_n(c, is_fourM=False):
    for ratio, den, num, a, b in cands:
        if n_mod_den(c, den, is_fourM) == 0:
            return (ratio, den, num, a, b)
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        ratio = Fraction(d - 1, d)
        return (ratio, d, d - 1, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            ratio = Fraction(d - 1, d)
            return (ratio, d, d - 1, None, None)
    return (Fraction(1, 1), None, None, None, None)
results = []
for idx, c in enumerate(cs):
    if idx == 1:
        res = minimal_ratio_for_n(c, is_fourM=True)
    else:
        res = minimal_ratio_for_n(c, is_fourM=False)
    results.append(res)
print(results)

### Turn 25
def n_mod(c, mod, is_fourM=False):
    M_mod = pow(3, A, mod)
    if is_fourM:
        return (5 * M_mod) % mod
    else:
        return (M_mod + c) % mod

mods = [5,7,11,13,17,23,29]
for c in [1848374, 10162574, 265710644, 44636594]:
    print('c',c, [n_mod(c, m) for m in mods])


### Turn 25
def n_mod(c, mod, is_fourM=False):
    M_mod = pow(3, A, mod)
    if is_fourM:
        return 5 * M_mod % mod
    else:
        return (M_mod + c) % mod
mods = [5, 7, 11, 13, 17, 23, 29]
for c in [1848374, 10162574, 265710644, 44636594]:
    print('c', c, [n_mod(c, m) for m in mods])

### Turn 26
def find_min_ratio_n(c, is_fourM=False, limit_a=30):
    # returns Fraction ratio, den, a,b
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab = a*b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            ratio = Fraction(num, den)
            # check divisibility: need den divides n and a|num and b|num
            if num % a != 0 or num % b != 0:
                continue
            if n_mod_den(c, den, is_fourM) == 0:
                # Additionally need N = (n//den)*num to be multiple of a and b, but that holds because num is multiple of a,b.
                return ratio, den, a, b
    # fallback: divisor method
    # compute s = n-1
    # check s % 4
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        return Fraction(3,4), d, None, None
    # else find smallest odd divisor
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            return Fraction(d-1, d), d, None, None
    # fallback
    return Fraction(1,1), None, None, None

# compute for each c
results2 = []
for idx,c in enumerate(cs):
    if idx == 1:
        res = find_min_ratio_n(c, is_fourM=True)
    else:
        res = find_min_ratio_n(c, is_fourM=False)
    results2.append(res)
results2


### Turn 26
def find_min_ratio_n(c, is_fourM=False, limit_a=30):
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            ratio = Fraction(num, den)
            if num % a != 0 or num % b != 0:
                continue
            if n_mod_den(c, den, is_fourM) == 0:
                return (ratio, den, a, b)
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        d = 4
        return (Fraction(3, 4), d, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            d = p
            return (Fraction(d - 1, d), d, None, None)
    return (Fraction(1, 1), None, None, None)
results2 = []
for idx, c in enumerate(cs):
    if idx == 1:
        res = find_min_ratio_n(c, is_fourM=True)
    else:
        res = find_min_ratio_n(c, is_fourM=False)
    results2.append(res)
print(results2)

### Turn 27
def find_min_ratio_n_sorted(c, is_fourM=False, limit_a=30):
    # generate candidates sorted by ratio
    cand = []
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab = a*b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            if num % a != 0 or num % b != 0:
                continue
            ratio = Fraction(num, den)
            cand.append((ratio, den, a, b))
    cand.sort(key=lambda x: x[0])  # ascending ratio
    for ratio, den, a, b in cand:
        if n_mod_den(c, den, is_fourM) == 0:
            return ratio, den, a, b
    # fallback to divisor method
    # compute s = n-1
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        return Fraction(3,4), 4, None, None
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            return Fraction(p-1, p), p, None, None
    return Fraction(1,1), None, None, None

results_sorted = []
for idx,c in enumerate(cs):
    if idx == 1:
        res = find_min_ratio_n_sorted(c, is_fourM=True)
    else:
        res = find_min_ratio_n_sorted(c, is_fourM=False)
    results_sorted.append(res)
results_sorted


### Turn 27
def find_min_ratio_n_sorted(c, is_fourM=False, limit_a=30):
    cand = []
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            if num % a != 0 or num % b != 0:
                continue
            ratio = Fraction(num, den)
            cand.append((ratio, den, a, b))
    cand.sort(key=lambda x: x[0])
    for ratio, den, a, b in cand:
        if n_mod_den(c, den, is_fourM) == 0:
            return (ratio, den, a, b)
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        return (Fraction(3, 4), 4, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            return (Fraction(p - 1, p), p, None, None)
    return (Fraction(1, 1), None, None, None)
results_sorted = []
for idx, c in enumerate(cs):
    if idx == 1:
        res = find_min_ratio_n_sorted(c, is_fourM=True)
    else:
        res = find_min_ratio_n_sorted(c, is_fourM=False)
    results_sorted.append(res)
print(results_sorted)

### Turn 28
def min_ratio_n(c, is_fourM=False, limit_a=30):
    best = None
    best_info = None
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab = a*b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            # check den divides n
            if n_mod_den(c, den, is_fourM) != 0:
                continue
            # compute required extra divisibility
            a1 = a // math.gcd(a, num)
            b1 = b // math.gcd(b, num)
            # need n divisible by den * a1 and den * b1
            if n_mod_den(c, den * a1, is_fourM) != 0:
                continue
            if n_mod_den(c, den * b1, is_fourM) != 0:
                continue
            ratio = Fraction(num, den)
            # keep minimal ratio
            if best is None or ratio < best:
                best = ratio
                best_info = (ratio, den, a, b, a1, b1)
    if best is not None:
        return best_info
    # fallback divisor method as before
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        return (Fraction(3,4), 4, None, None, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            return (Fraction(p-1, p), p, None, None, None, None)
    return (Fraction(1,1), None, None, None, None, None)

# compute for each c
results_final = []
for idx,c in enumerate(cs):
    if idx == 1:
        res = min_ratio_n(c, is_fourM=True)
    else:
        res = min_ratio_n(c, is_fourM=False)
    results_final.append(res)
results_final


### Turn 28
def min_ratio_n(c, is_fourM=False, limit_a=30):
    best = None
    best_info = None
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            if n_mod_den(c, den, is_fourM) != 0:
                continue
            a1 = a // math.gcd(a, num)
            b1 = b // math.gcd(b, num)
            if n_mod_den(c, den * a1, is_fourM) != 0:
                continue
            if n_mod_den(c, den * b1, is_fourM) != 0:
                continue
            ratio = Fraction(num, den)
            if best is None or ratio < best:
                best = ratio
                best_info = (ratio, den, a, b, a1, b1)
    if best is not None:
        return best_info
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        return (Fraction(3, 4), 4, None, None, None, None)
    for p in sp.primerange(3, 5000):
        M_mod_p = pow(3, A, p)
        if is_fourM:
            s_mod_p = (5 * M_mod_p - 1) % p
        else:
            s_mod_p = (M_mod_p + c - 1) % p
        if s_mod_p == 0:
            return (Fraction(p - 1, p), p, None, None, None, None)
    return (Fraction(1, 1), None, None, None, None, None)
results_final = []
for idx, c in enumerate(cs):
    if idx == 1:
        res = min_ratio_n(c, is_fourM=True)
    else:
        res = min_ratio_n(c, is_fourM=False)
    results_final.append(res)
print(results_final)

### Turn 29
def n_mod(c, mod, is_fourM=False):
    M_mod = pow(3, A, mod)
    if is_fourM:
        return (5 * M_mod) % mod
    else:
        return (M_mod + c) % mod

print(n_mod(1848374, 25))


### Turn 29
def n_mod(c, mod, is_fourM=False):
    M_mod = pow(3, A, mod)
    if is_fourM:
        return 5 * M_mod % mod
    else:
        return (M_mod + c) % mod
print(n_mod(1848374, 25))

### Turn 30
print(n_mod(10162574, 47))


### Turn 30
print(n_mod(10162574, 47))

### Turn 31
print(n_mod(265710644, 103))


### Turn 31
print(n_mod(265710644, 103))

### Turn 32
print(n_mod(44636594, 103))


### Turn 32
print(n_mod(44636594, 103))

### Turn 33
from math import floor
A = math.factorial(2025)
pairs_info = [
    (2,3),   # c=0 and 4M
    (2,3),
    (16,25),
    (30,47),
    (75,103),
    (75,103)
]

total = sum((A * num) // den for num, den in pairs_info)
p = total
q = A
g = math.gcd(p, q)
p_red = p // g
q_red = q // g
(p_red + q_red) % 99991


### Turn 33
from math import floor
A = math.factorial(2025)
pairs_info = [(2, 3), (2, 3), (16, 25), (30, 47), (75, 103), (75, 103)]
total = sum((A * num // den for num, den in pairs_info))
p = total
q = A
g = math.gcd(p, q)
p_red = p // g
q_red = q // g
print((p_red + q_red) % 99991)

### Turn 34
def brute_f(n, maxN=2000):
    # brute compute f(n) by searching N up to maxN
    for N in range(1, maxN+1):
        # get divisors of N
        divs = [d for d in range(1, N+1) if N % d == 0]
        L = len(divs)
        for i in range(L):
            for j in range(i+1, L):
                for k in range(j+1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
    return None

def compute_f_algorithm(n, M):
    # compute using our algorithm: find minimal ratio via candidates
    # We'll generate candidates as before up to a,b limit maybe 30
    limit_a=30
    best_ratio = None
    best_N = None
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab = a*b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            # need den divides n
            if n % den != 0:
                continue
            t = n // den
            # check N = t * num is divisible by a and b
            N = t * num
            if N % a != 0 or N % b != 0:
                continue
            if best_N is None or N < best_N:
                best_N = N
                best_ratio = Fraction(num, den)
    # fallback divisor method
    if best_N is None:
        s = n - 1
        # find smallest divisor >2
        if s % 4 == 0:
            d = 4
        else:
            # find smallest odd divisor >2
            d = None
            for p in sp.primerange(3, 500):
                if s % p == 0:
                    d = p
                    break
            if d is None:
                # s prime
                # use a=2,b=s-2
                N = 2 * (s - 2)
                return N
        N = s - s // d
        return N
    return best_N

# test for e=5
e = 5
M_small = 3**e
c_vals_test = [0, 4*M_small, 1848374 % (M_small*10), 10162574 % (M_small*10), 265710644 % (M_small*10), 44636594 % (M_small*10)]
# Use small c's maybe unrealistic; but we just test few random c's
test_cs = [0, 4*M_small, 10, 12, 14]  # small examples
for c in test_cs:
    n = M_small + c
    f_br = brute_f(n, maxN=2000)
    f_alg = compute_f_algorithm(n, M_small)
    print("c",c,"n",n,"brute",f_br,"alg",f_alg)


### Turn 34
def brute_f(n, maxN=2000):
    for N in range(1, maxN + 1):
        divs = [d for d in range(1, N + 1) if N % d == 0]
        L = len(divs)
        for i in range(L):
            for j in range(i + 1, L):
                for k in range(j + 1, L):
                    if divs[i] + divs[j] + divs[k] == n:
                        return N
    return None

def compute_f_algorithm(n, M):
    limit_a = 30
    best_ratio = None
    best_N = None
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            if n % den != 0:
                continue
            t = n // den
            N = t * num
            if N % a != 0 or N % b != 0:
                continue
            if best_N is None or N < best_N:
                best_N = N
                best_ratio = Fraction(num, den)
    if best_N is None:
        s = n - 1
        if s % 4 == 0:
            d = 4
        else:
            d = None
            for p in sp.primerange(3, 500):
                if s % p == 0:
                    d = p
                    break
            if d is None:
                N = 2 * (s - 2)
                return N
        N = s - s // d
        return N
    return best_N
e = 5
M_small = 3 ** e
c_vals_test = [0, 4 * M_small, 1848374 % (M_small * 10), 10162574 % (M_small * 10), 265710644 % (M_small * 10), 44636594 % (M_small * 10)]
test_cs = [0, 4 * M_small, 10, 12, 14]
for c in test_cs:
    n = M_small + c
    f_br = brute_f(n, maxN=2000)
    f_alg = compute_f_algorithm(n, M_small)
    print('c', c, 'n', n, 'brute', f_br, 'alg', f_alg)

### Turn 35
candidates_den5 = []
for a in range(2,100):
    for b in range(a+1,100):
        ab=a*b
        D=ab+a+b
        g=math.gcd(ab,D)
        den = D//g
        if den==5:
            num = ab//g
            candidates_den5.append((a,b,num))
candidates_den5[:10], len(candidates_den5)


### Turn 35
candidates_den5 = []
for a in range(2, 100):
    for b in range(a + 1, 100):
        ab = a * b
        D = ab + a + b
        g = math.gcd(ab, D)
        den = D // g
        if den == 5:
            num = ab // g
            candidates_den5.append((a, b, num))
print((candidates_den5[:10], len(candidates_den5)))

### Turn 36
def ratio_ab(a,b):
    ab=a*b
    D=ab+a+b
    g=math.gcd(ab,D)
    return Fraction(ab//g, D//g)

def test_ratios(limit):
    ratios = []
    for a in range(2,limit+1):
        for b in range(a+1,limit+1):
            ratios.append((ratio_ab(a,b), a,b))
    ratios.sort(key=lambda x:x[0])
    return ratios[:10]

test_ratios(30)[:10]


### Turn 36
def ratio_ab(a, b):
    ab = a * b
    D = ab + a + b
    g = math.gcd(ab, D)
    return Fraction(ab // g, D // g)

def test_ratios(limit):
    ratios = []
    for a in range(2, limit + 1):
        for b in range(a + 1, limit + 1):
            ratios.append((ratio_ab(a, b), a, b))
    ratios.sort(key=lambda x: x[0])
    return ratios[:10]
print(test_ratios(30)[:10])

### Turn 37
from fractions import Fraction
A = math.factorial(2025)
terms = [
    Fraction((A*2)//3, A),  # c=0 ratio 2/3
    Fraction((A*2)//3, A),  # c=4M ratio 2/3
    Fraction((A*16)//25, A),  # c=1848374 ratio 16/25
    Fraction((A*30)//47, A),  # c=10162574 ratio 30/47
    Fraction((A*75)//103, A),  # c=265710644 ratio 75/103
    Fraction((A*75)//103, A)   # c=44636594 ratio 75/103
]
S = sum(terms, Fraction(0,1))
S, S.numerator, S.denominator


### Turn 37
from fractions import Fraction
A = math.factorial(2025)
terms = [Fraction(A * 2 // 3, A), Fraction(A * 2 // 3, A), Fraction(A * 16 // 25, A), Fraction(A * 30 // 47, A), Fraction(A * 75 // 103, A), Fraction(A * 75 // 103, A)]
S = sum(terms, Fraction(0, 1))
print((S, S.numerator, S.denominator))

### Turn 38
import math
math.gcd(1476968,363075)


### Turn 38
import math
print(math.gcd(1476968, 363075))

### Turn 39
(p+q) % 99991


### Turn 39
print((p + q) % 99991)

### Turn 40
from fractions import Fraction
S_exact = Fraction(2,3)+Fraction(2,3)+Fraction(16,25)+Fraction(30,47)+Fraction(75,103)+Fraction(75,103)
S_exact, S_exact.numerator, S_exact.denominator


### Turn 40
from fractions import Fraction
S_exact = Fraction(2, 3) + Fraction(2, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(75, 103) + Fraction(75, 103)
print((S_exact, S_exact.numerator, S_exact.denominator))

### Turn 41
1476968+363075


### Turn 41
print(1476968 + 363075)

### Turn 42
1840043 % 99991


### Turn 42
print(1840043 % 99991)

### Turn 43
M_mod_7 = pow(3, A, 7)
M_mod_7


### Turn 43
M_mod_7 = pow(3, A, 7)
print(M_mod_7)

### Turn 44
def find_best_ratio_bruteforce(c, is_fourM=False, limit_a=100):
    best_ratio = None
    best_info = None
    for a in range(2, limit_a+1):
        for b in range(a+1, limit_a+1):
            ab=a*b
            D=ab+a+b
            g=math.gcd(ab,D)
            num=ab//g
            den=D//g
            # condition den divides n
            if n_mod_den(c, den, is_fourM) != 0:
                continue
            t = (n_mod_den(c, den, is_fourM) == 0)  # placeholder
            # need n divisible by den*a1 and den*b1
            a1 = a // math.gcd(a, num)
            b1 = b // math.gcd(b, num)
            if n_mod_den(c, den*a1, is_fourM) != 0:
                continue
            if n_mod_den(c, den*b1, is_fourM) != 0:
                continue
            ratio = Fraction(num, den)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_info = (a,b,num,den)
    # also consider divisor method
    # compute s = n-1
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        ratio = Fraction(3,4)
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
    else:
        # find smallest odd prime divisor of s
        for p in sp.primerange(3, 5000):
            M_mod_p = pow(3, A, p)
            if is_fourM:
                s_mod_p = (5 * M_mod_p - 1) % p
            else:
                s_mod_p = (M_mod_p + c - 1) % p
            if s_mod_p == 0:
                ratio = Fraction(p-1, p)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                break
    return best_ratio, best_info

for idx,c in enumerate(cs):
    if idx==1:
        r,info = find_best_ratio_bruteforce(c, is_fourM=True)
    else:
        r,info = find_best_ratio_bruteforce(c)
    print(idx, c, r, info)


### Turn 44
def find_best_ratio_bruteforce(c, is_fourM=False, limit_a=100):
    best_ratio = None
    best_info = None
    for a in range(2, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            ab = a * b
            D = ab + a + b
            g = math.gcd(ab, D)
            num = ab // g
            den = D // g
            if n_mod_den(c, den, is_fourM) != 0:
                continue
            t = n_mod_den(c, den, is_fourM) == 0
            a1 = a // math.gcd(a, num)
            b1 = b // math.gcd(b, num)
            if n_mod_den(c, den * a1, is_fourM) != 0:
                continue
            if n_mod_den(c, den * b1, is_fourM) != 0:
                continue
            ratio = Fraction(num, den)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_info = (a, b, num, den)
    M_mod_4 = pow(3, A, 4)
    if is_fourM:
        s_mod_4 = (5 * M_mod_4 - 1) % 4
    else:
        s_mod_4 = (M_mod_4 + c - 1) % 4
    if s_mod_4 == 0:
        ratio = Fraction(3, 4)
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
    else:
        for p in sp.primerange(3, 5000):
            M_mod_p = pow(3, A, p)
            if is_fourM:
                s_mod_p = (5 * M_mod_p - 1) % p
            else:
                s_mod_p = (M_mod_p + c - 1) % p
            if s_mod_p == 0:
                ratio = Fraction(p - 1, p)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                break
    return (best_ratio, best_info)
for idx, c in enumerate(cs):
    if idx == 1:
        r, info = find_best_ratio_bruteforce(c, is_fourM=True)
    else:
        r, info = find_best_ratio_bruteforce(c)
    print(idx, c, r, info)

