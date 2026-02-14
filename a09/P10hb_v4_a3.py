# Auto-extracted from: P10hb_v4.md
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



### Turn 0
import sympy as sp

def f_of_n(n):
    # search m from 1 up to n (since m must be less than n)
    for m in range(1, n):
        # find divisors of m
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        # choose any 3 distinct divisors
        # can use combinations
        for a in range(len(divs)):
            for b in range(a+1, len(divs)):
                for c in range(b+1, len(divs)):
                    s = divs[a] + divs[b] + divs[c]
                    if s == n:
                        return m
    return None

# compute small values
vals = {}
for n in range(6, 101):
    vals[n] = f_of_n(n)
# Print some
list(vals.items())[:20]


### Turn 0
import sympy as sp

def f_of_n(n):
    for m in range(1, n):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a in range(len(divs)):
            for b in range(a + 1, len(divs)):
                for c in range(b + 1, len(divs)):
                    s = divs[a] + divs[b] + divs[c]
                    if s == n:
                        return m
    return None
vals = {}
for n in range(6, 101):
    vals[n] = f_of_n(n)
print(list(vals.items())[:20])

### Turn 1
def f_of_n_bruteforce(n, limit=500):
    best = None
    for m in range(1, limit+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        # check any combination of 3 distinct divisors
        from itertools import combinations
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                if best is None or m < best:
                    best = m
    return best

for n in range(6, 31):
    f = f_of_n_bruteforce(n, limit=200)
    print(n, f)


### Turn 1
def f_of_n_bruteforce(n, limit=500):
    best = None
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        from itertools import combinations
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                if best is None or m < best:
                    best = m
    return best
for n in range(6, 31):
    f = f_of_n_bruteforce(n, limit=200)
    print(n, f)

### Turn 2
def f_of_n(n, limit=2000):
    best = None
    for m in range(1, limit+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        # try all unordered triples
        for a in range(len(divs)):
            for b in range(a+1, len(divs)):
                for c in range(b+1, len(divs)):
                    if divs[a] + divs[b] + divs[c] == n:
                        if best is None or m < best:
                            best = m
        # early break if best found and m beyond best? not needed
    return best

# compute for n up to 100
res = {}
for n in range(6, 101):
    res[n] = f_of_n(n, limit=200)
print({n:res[n] for n in range(6, 31)})


### Turn 2
def f_of_n(n, limit=2000):
    best = None
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a in range(len(divs)):
            for b in range(a + 1, len(divs)):
                for c in range(b + 1, len(divs)):
                    if divs[a] + divs[b] + divs[c] == n:
                        if best is None or m < best:
                            best = m
    return best
res = {}
for n in range(6, 101):
    res[n] = f_of_n(n, limit=200)
print({n: res[n] for n in range(6, 31)})

### Turn 3
def smallest_composite_ge(k):
    # composite numbers with at least three divisors: 4,6,8,9,10,...
    # but note 4 has divisors 1,2,4 (only 3 distinct). 4 qualifies.
    n = k
    while True:
        if n > 1 and not sp.isprime(n):
            # check at least 3 divisors distinct: all composite n have at least 3 divisors? Not all: squares of prime have 3 divisors (e.g., 9). That's ok.
            return n
        n += 1

def f_estimate(n):
    return smallest_composite_ge((n+1)//2)

for n in range(6, 31):
    print(n, f_estimate(n), res[n])


### Turn 3
def smallest_composite_ge(k):
    n = k
    while True:
        if n > 1 and (not sp.isprime(n)):
            return n
        n += 1

def f_estimate(n):
    return smallest_composite_ge((n + 1) // 2)
for n in range(6, 31):
    print(n, f_estimate(n), res[n])

### Turn 4
def f_bruteforce_limit(n, limit_factor=2):
    # search m up to some factor of n maybe.
    limit = int(limit_factor * n) + 100
    best = None
    for m in range(1, limit+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a in range(len(divs)):
            for b in range(a+1, len(divs)):
                for c in range(b+1, len(divs)):
                    if divs[a] + divs[b] + divs[c] == n:
                        if best is None or m < best:
                            best = m
        # optional early exit: if best found and m exceed best, break
        if best is not None and m > best:
            break
    return best

def compute_f_for_n_values(base, cs):
    res = {}
    for c in cs:
        n = base + c
        res[c] = f_bruteforce_limit(n, limit_factor=1)  # maybe n itself enough
    return res

base = 3**5
cs = [0,1,2,3,4,5,6,7,10,20]
print(base, compute_f_for_n_values(base, cs))


### Turn 4
def f_bruteforce_limit(n, limit_factor=2):
    limit = int(limit_factor * n) + 100
    best = None
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a in range(len(divs)):
            for b in range(a + 1, len(divs)):
                for c in range(b + 1, len(divs)):
                    if divs[a] + divs[b] + divs[c] == n:
                        if best is None or m < best:
                            best = m
        if best is not None and m > best:
            break
    return best

def compute_f_for_n_values(base, cs):
    res = {}
    for c in cs:
        n = base + c
        res[c] = f_bruteforce_limit(n, limit_factor=1)
    return res
base = 3 ** 5
cs = [0, 1, 2, 3, 4, 5, 6, 7, 10, 20]
print(base, compute_f_for_n_values(base, cs))

### Turn 5
for k in range(4,7):
    base = 3**k
    print(k, base, f_bruteforce_limit(base, limit_factor=1))


### Turn 5
for k in range(4, 7):
    base = 3 ** k
    print(k, base, f_bruteforce_limit(base, limit_factor=1))

### Turn 6
import math, sys
from math import factorial

K = math.factorial(2025)  # huge int
# Precompute M_mod_d for needed d range on the fly

def smallest_odd_divisor_ge7_of_n(c, is_multiple_of_M=False, multiple_factor=1):
    # n = M + c if not multiple_of_M else multiple_factor * M + c? Actually for 4M case, c = 4M, so n = M + c = 5M
    # We'll handle specific case outside.
    # But for general use, provide c and a flag for adding c mod d.
    pass


### Turn 6
import math, sys
from math import factorial
K = math.factorial(2025)

def smallest_odd_divisor_ge7_of_n(c, is_multiple_of_M=False, multiple_factor=1):
    pass

### Turn 7
def smallest_odd_divisor_ge7(M_mod_cache, n_mod_func, max_search=20000):
    # n_mod_func(d) returns n % d
    for d in range(7, max_search+1, 2):
        if n_mod_func(d) == 0:
            return d
    return None

# Compute M_mod_d for many d
def M_mod(d):
    return pow(3, K, d)

# Let's test for c=0 case: n = M
def n_mod_M(d):
    return M_mod(d) % d

k0 = smallest_odd_divisor_ge7(None, n_mod_M, max_search=2000)
print("k0:", k0)


### Turn 7
def smallest_odd_divisor_ge7(M_mod_cache, n_mod_func, max_search=20000):
    for d in range(7, max_search + 1, 2):
        if n_mod_func(d) == 0:
            return d
    return None

def M_mod(d):
    return pow(3, K, d)

def n_mod_M(d):
    return M_mod(d) % d
k0 = smallest_odd_divisor_ge7(None, n_mod_M, max_search=2000)
print('k0:', k0)

### Turn 8
def n_mod_with_c(c):
    # returns a function that computes n_mod_d = (M_mod(d) + c) % d
    def f(d):
        return (pow(3, K, d) + c) % d
    return f

cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    k = smallest_odd_divisor_ge7(None, n_mod_with_c(c), max_search=5000)
    print("c:", c, "k:", k)


### Turn 8
def n_mod_with_c(c):

    def f(d):
        return (pow(3, K, d) + c) % d
    return f
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    k = smallest_odd_divisor_ge7(None, n_mod_with_c(c), max_search=5000)
    print('c:', c, 'k:', k)

### Turn 9
import math, sys
fact2025 = math.factorial(2025)
def X(k):
    return (fact2025 // k) * (k-3)

for k in [9,25,97,103]:
    print(k, fact2025 % k, X(k) % 1)


### Turn 9
import math, sys
fact2025 = math.factorial(2025)

def X(k):
    return fact2025 // k * (k - 3)
for k in [9, 25, 97, 103]:
    print(k, fact2025 % k, X(k) % 1)

### Turn 10
from fractions import Fraction

S = Fraction(2,3) + Fraction(10,3) + Fraction(22,25) + Fraction(22,25) + Fraction(94,97) + Fraction(100,103)
S


### Turn 10
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(22, 25) + Fraction(22, 25) + Fraction(94, 97) + Fraction(100, 103)
print(S)

### Turn 11
p = S.numerator
q = S.denominator
mod = 99991
(p+q) % mod


### Turn 11
p = S.numerator
q = S.denominator
mod = 99991
print((p + q) % mod)

### Turn 12
def n_mod(d, c):
    return (pow(3, K, d) + c) % d

for c in [1848374, 10162574, 265710644, 44636594]:
    for d in [7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45]:
        if n_mod(d,c) == 0:
            print("c",c,"divisible by",d)
            break


### Turn 12
def n_mod(d, c):
    return (pow(3, K, d) + c) % d
for c in [1848374, 10162574, 265710644, 44636594]:
    for d in [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]:
        if n_mod(d, c) == 0:
            print('c', c, 'divisible by', d)
            break

### Turn 13
def find_smallest_divisor(c, limit=2000):
    for d in range(7, limit+1, 2):
        if n_mod(d,c)==0:
            return d
    return None

for c in [265710644, 44636594]:
    print(c, find_smallest_divisor(c, limit=2000))


### Turn 13
def find_smallest_divisor(c, limit=2000):
    for d in range(7, limit + 1, 2):
        if n_mod(d, c) == 0:
            return d
    return None
for c in [265710644, 44636594]:
    print(c, find_smallest_divisor(c, limit=2000))

### Turn 14
import math
math.gcd(p,q)


### Turn 14
import math
print(math.gcd(p, q))

### Turn 15
from fractions import Fraction

def g_from_k(k, factor=1):
    tmax = (k-1)//3
    num = k - tmax - 1
    return Fraction(num, k) * factor

# c values
g0 = g_from_k(9, factor=1)  # c=0
g4M = g_from_k(9, factor=5)  # c=4M factor 5
g1 = g_from_k(25, factor=1)  # c=1848374
g2 = g_from_k(25, factor=1)  # c=10162574
g3 = g_from_k(97, factor=1)
g4 = g_from_k(103, factor=1)

S2 = g0 + g4M + g1 + g2 + g3 + g4
S2


### Turn 15
from fractions import Fraction

def g_from_k(k, factor=1):
    tmax = (k - 1) // 3
    num = k - tmax - 1
    return Fraction(num, k) * factor
g0 = g_from_k(9, factor=1)
g4M = g_from_k(9, factor=5)
g1 = g_from_k(25, factor=1)
g2 = g_from_k(25, factor=1)
g3 = g_from_k(97, factor=1)
g4 = g_from_k(103, factor=1)
S2 = g0 + g4M + g1 + g2 + g3 + g4
print(S2)

### Turn 16
(p:=1648512+249775) % 99991


### Turn 16
print((p := (1648512 + 249775)) % 99991)

### Turn 17
from math import gcd
num = 1648512
den = 249775
gcd(num, den)


### Turn 17
from math import gcd
num = 1648512
den = 249775
print(gcd(num, den))

### Turn 18
def f_formula(n):
    # find smallest odd divisor k >=7 dividing n
    for k in range(7, 1000, 2):
        if n % k == 0:
            tmax = (k-1)//3
            return n * (k - tmax - 1) // k
    # if not found, need other method: use brute maybe
    return None

# test up to 200
bad = []
for n in range(6, 201):
    brute = f_bruteforce_limit(n, limit=500)  # compute f via brute up to maybe 500 or more
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
bad[:20], len(bad)


### Turn 18
def f_formula(n):
    for k in range(7, 1000, 2):
        if n % k == 0:
            tmax = (k - 1) // 3
            return n * (k - tmax - 1) // k
    return None
bad = []
for n in range(6, 201):
    brute = f_bruteforce_limit(n, limit=500)
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
print((bad[:20], len(bad)))

### Turn 19
def brute_f(n, max_m=2000):
    best = None
    for m in range(1, max_m+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        # check combos
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                if best is None or m < best:
                    best = m
        # early break if best found and m exceeds best? but we just continue
    return best

bad = []
for n in range(6, 151):
    brute = brute_f(n, max_m=500)  # guess enough
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
bad[:20], len(bad)


### Turn 19
def brute_f(n, max_m=2000):
    best = None
    for m in range(1, max_m + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                if best is None or m < best:
                    best = m
    return best
bad = []
for n in range(6, 151):
    brute = brute_f(n, max_m=500)
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
print((bad[:20], len(bad)))

### Turn 20
from itertools import combinations
def brute_f(n, max_m=2000):
    best = None
    for m in range(1, max_m+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                if best is None or m < best:
                    best = m
        if best is not None and m > best:
            break
    return best

bad = []
for n in range(6, 151):
    brute = brute_f(n, max_m=500)
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
len(bad), bad[:20]


### Turn 20
from itertools import combinations

def brute_f(n, max_m=2000):
    best = None
    for m in range(1, max_m + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                if best is None or m < best:
                    best = m
        if best is not None and m > best:
            break
    return best
bad = []
for n in range(6, 151):
    brute = brute_f(n, max_m=500)
    formula = f_formula(n)
    if brute != formula:
        bad.append((n, brute, formula))
print((len(bad), bad[:20]))

### Turn 21
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 6)


### Turn 21
cs = [0, 1848374, 10162574, 265710644, 44636594]
for c in cs:
    print(c, c % 6)

### Turn 22
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25)*2 + Fraction(64,97) + Fraction(68,103)
S, S.numerator, S.denominator


### Turn 22
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) * 2 + Fraction(64, 97) + Fraction(68, 103)
print((S, S.numerator, S.denominator))

### Turn 23
(p+q) % 99991


### Turn 23
print((p + q) % 99991)

### Turn 24
p=1648512
q=249775
p+q


### Turn 24
p = 1648512
q = 249775
print(p + q)

### Turn 25
1898287 % 99991


### Turn 25
print(1898287 % 99991)

### Turn 26
def smallest_odd_divisor_ge7_bruteforce(c, limit=5000):
    for d in range(7, limit+1, 2):
        if (pow(3, K, d) + c) % d == 0:
            return d
    return None

for c in cs:
    print(c, smallest_odd_divisor_ge7_bruteforce(c, limit=2000))


### Turn 26
def smallest_odd_divisor_ge7_bruteforce(c, limit=5000):
    for d in range(7, limit + 1, 2):
        if (pow(3, K, d) + c) % d == 0:
            return d
    return None
for c in cs:
    print(c, smallest_odd_divisor_ge7_bruteforce(c, limit=2000))

### Turn 27
def brute_f_limited(n, max_m=2000):
    best = None
    for m in range(1, max_m+1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a,b,c in combinations(divs,3):
            if a+b+c == n:
                if best is None or m < best:
                    best = m
        if best is not None and m > best:
            break
    return best

print(brute_f_limited(275, max_m=500))


### Turn 27
def brute_f_limited(n, max_m=2000):
    best = None
    for m in range(1, max_m + 1):
        divs = sp.divisors(m)
        if len(divs) < 3:
            continue
        for a, b, c in combinations(divs, 3):
            if a + b + c == n:
                if best is None or m < best:
                    best = m
        if best is not None and m > best:
            break
    return best
print(brute_f_limited(275, max_m=500))

### Turn 28
def find_triple(n,m):
    divs = sp.divisors(m)
    for a,b,c in combinations(divs,3):
        if a+b+c == n:
            return a,b,c
    return None

triple = find_triple(275,150)
triple


### Turn 28
def find_triple(n, m):
    divs = sp.divisors(m)
    for a, b, c in combinations(divs, 3):
        if a + b + c == n:
            return (a, b, c)
    return None
triple = find_triple(275, 150)
print(triple)

### Turn 29
def best_ratio_for_n(c, include_multiple_factor=1):
    # n = M + c; include factor for multiples like c=4M we handle factor 5.
    # We'll treat n_mod_d function.
    # For c where c is multiple of M (4M) we have n = (1+factor)*M
    # Provide factor argument: for c=4M factor=5
    # For other c, factor=1
    factor = include_multiple_factor
    # function to compute n mod d
    def n_mod(d):
        # compute M_mod = 3^K mod d
        M_mod = pow(3, K, d)
        return (factor * M_mod + c) % d
    # enumerate pairs p<q up to limit
    best = None
    best_pair = None
    limit = 50
    for p in range(2, limit+1):
        for q in range(p+1, limit+1):
            D = p*q + p + q
            # check if D divides n
            if n_mod(D) == 0:
                # compute m = n * p*q / D = (M + c) * p*q / D
                # ratio = p*q / D
                ratio = Fraction(p*q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p,q,D)
    return best, best_pair

# test for each c
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    ratio, pair = best_ratio_for_n(c)
    print("c",c,"best ratio", ratio, "pair", pair)


### Turn 29
def best_ratio_for_n(c, include_multiple_factor=1):
    factor = include_multiple_factor

    def n_mod(d):
        M_mod = pow(3, K, d)
        return (factor * M_mod + c) % d
    best = None
    best_pair = None
    limit = 50
    for p in range(2, limit + 1):
        for q in range(p + 1, limit + 1):
            D = p * q + p + q
            if n_mod(D) == 0:
                ratio = Fraction(p * q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p, q, D)
    return (best, best_pair)
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    ratio, pair = best_ratio_for_n(c)
    print('c', c, 'best ratio', ratio, 'pair', pair)

### Turn 30
def find_ratio_target(target_num, target_den, limit=100):
    solutions = []
    for p in range(2, limit+1):
        for q in range(p+1, limit+1):
            if p*q * target_den == target_num * (p*q + p + q):
                solutions.append((p,q))
    return solutions

find_ratio_target(16,25,limit=100)


### Turn 30
def find_ratio_target(target_num, target_den, limit=100):
    solutions = []
    for p in range(2, limit + 1):
        for q in range(p + 1, limit + 1):
            if p * q * target_den == target_num * (p * q + p + q):
                solutions.append((p, q))
    return solutions
print(find_ratio_target(16, 25, limit=100))

### Turn 31
def M_mod(d):
    return pow(3, K, d)

M_mod_50 = M_mod(50)
M_mod_50


### Turn 31
def M_mod(d):
    return pow(3, K, d)
M_mod_50 = M_mod(50)
print(M_mod_50)

### Turn 32
c = 1848374
n_mod_50 = (M_mod_50 + c) % 50
n_mod_50


### Turn 32
c = 1848374
n_mod_50 = (M_mod_50 + c) % 50
print(n_mod_50)

### Turn 33
def best_ratio_by_enumeration(c, factor=1, limit=200):
    # n = factor*M + c
    def n_mod(d):
        return (factor * M_mod(d) + c) % d
    best_ratio = None
    best_pair = None
    for p in range(2, limit+1):
        for q in range(p+1, limit+1):
            D = p*q + p + q
            if n_mod(D) == 0:
                ratio = Fraction(p*q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p,q,D)
    return best_ratio, best_pair

for c in [1848374, 10162574]:
    r,pair = best_ratio_by_enumeration(c, limit=200)
    print(c, r, pair)


### Turn 33
def best_ratio_by_enumeration(c, factor=1, limit=200):

    def n_mod(d):
        return (factor * M_mod(d) + c) % d
    best_ratio = None
    best_pair = None
    for p in range(2, limit + 1):
        for q in range(p + 1, limit + 1):
            D = p * q + p + q
            if n_mod(D) == 0:
                ratio = Fraction(p * q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p, q, D)
    return (best_ratio, best_pair)
for c in [1848374, 10162574]:
    r, pair = best_ratio_by_enumeration(c, limit=200)
    print(c, r, pair)

### Turn 34
# Precompute M_mod for a set of moduli
def precompute_M_mod(moduli):
    M_mods = {}
    for d in moduli:
        M_mods[d] = pow(3, K, d)
    return M_mods

# generate list of D values for p,q up to 200
moduli_set = set()
limit = 200
pairs = []
for p in range(2, limit+1):
    for q in range(p+1, limit+1):
        D = p*q + p + q
        moduli_set.add(D)
pairs = [(p,q,p*q + p + q) for p in range(2, limit+1) for q in range(p+1, limit+1)]

M_mods = precompute_M_mod(moduli_set)
len(M_mods)


### Turn 34
def precompute_M_mod(moduli):
    M_mods = {}
    for d in moduli:
        M_mods[d] = pow(3, K, d)
    return M_mods
moduli_set = set()
limit = 200
pairs = []
for p in range(2, limit + 1):
    for q in range(p + 1, limit + 1):
        D = p * q + p + q
        moduli_set.add(D)
pairs = [(p, q, p * q + p + q) for p in range(2, limit + 1) for q in range(p + 1, limit + 1)]
M_mods = precompute_M_mod(moduli_set)
print(len(M_mods))

### Turn 35
def best_ratio_for_n(c, factor=1):
    best_ratio = None
    best_pair = None
    for p,q,D in pairs:
        n_mod = (factor * M_mods[D] + c) % D
        if n_mod == 0:
            ratio = Fraction(p*q, D)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_pair = (p,q,D)
    return best_ratio, best_pair

# compute for each c
for c in [0, 1848374, 10162574, 265710644, 44636594]:
    ratio, pair = best_ratio_for_n(c)
    print("c:",c,"ratio:",ratio,"pair:",pair)


### Turn 35
def best_ratio_for_n(c, factor=1):
    best_ratio = None
    best_pair = None
    for p, q, D in pairs:
        n_mod = (factor * M_mods[D] + c) % D
        if n_mod == 0:
            ratio = Fraction(p * q, D)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_pair = (p, q, D)
    return (best_ratio, best_pair)
for c in [0, 1848374, 10162574, 265710644, 44636594]:
    ratio, pair = best_ratio_for_n(c)
    print('c:', c, 'ratio:', ratio, 'pair:', pair)

### Turn 36
def n_mod_d(c, d, factor=1):
    return (factor * pow(3, K, d) + c) % d

for c in [0, 1848374, 10162574, 265710644, 44636594]:
    print("c",c,"n mod 11:", n_mod_d(c, 11))


### Turn 36
def n_mod_d(c, d, factor=1):
    return (factor * pow(3, K, d) + c) % d
for c in [0, 1848374, 10162574, 265710644, 44636594]:
    print('c', c, 'n mod 11:', n_mod_d(c, 11))

### Turn 37
for c in [1848374]:
    print(n_mod_d(c, 53))


### Turn 37
for c in [1848374]:
    print(n_mod_d(c, 53))

### Turn 38
print(n_mod_d(1848374, 47))


### Turn 38
print(n_mod_d(1848374, 47))

### Turn 39
solutions = []
for p in range(2, 50):
    for q in range(p+1, 50):
        if p*q + p + q == 25:
            solutions.append((p,q))
solutions


### Turn 39
solutions = []
for p in range(2, 50):
    for q in range(p + 1, 50):
        if p * q + p + q == 25:
            solutions.append((p, q))
print(solutions)

### Turn 40
from fractions import Fraction
r1 = Fraction(16,25)
r2 = Fraction(30,47)
float(r1), float(r2)


### Turn 40
from fractions import Fraction
r1 = Fraction(16, 25)
r2 = Fraction(30, 47)
print((float(r1), float(r2)))

### Turn 41
r1 = Fraction(64,97)
r2 = Fraction(130,197)
float(r1), float(r2)


### Turn 41
r1 = Fraction(64, 97)
r2 = Fraction(130, 197)
print((float(r1), float(r2)))

### Turn 42
r1 = Fraction(68,103)
r2 = Fraction(110,167)
float(r1), float(r2)


### Turn 42
r1 = Fraction(68, 103)
r2 = Fraction(110, 167)
print((float(r1), float(r2)))

### Turn 43
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S, S.numerator, S.denominator


### Turn 43
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print((S, S.numerator, S.denominator))

### Turn 44
(p+q) % 99991


### Turn 44
print((p + q) % 99991)

### Turn 45
c=1848374
print(n_mod_d(c,14))


### Turn 45
c = 1848374
print(n_mod_d(c, 14))

### Turn 46
print(n_mod_d(c,17))


### Turn 46
print(n_mod_d(c, 17))

### Turn 47
print(n_mod_d(c,20))


### Turn 47
print(n_mod_d(c, 20))

### Turn 48
print(n_mod_d(c,23))


### Turn 48
print(n_mod_d(c, 23))

### Turn 49
print(n_mod_d(c,26))


### Turn 49
print(n_mod_d(c, 26))

### Turn 50
print(n_mod_d(c,29))


### Turn 50
print(n_mod_d(c, 29))

### Turn 51
print(n_mod_d(c,32))


### Turn 51
print(n_mod_d(c, 32))

### Turn 52
c = 10162574
print("n mod 41:", n_mod_d(c,41))


### Turn 52
c = 10162574
print('n mod 41:', n_mod_d(c, 41))

### Turn 53
print(n_mod_d(c,44))


### Turn 53
print(n_mod_d(c, 44))

### Turn 54
print(n_mod_d(c,38))


### Turn 54
print(n_mod_d(c, 38))

### Turn 55
def best_ratio_for_n_general(c, factor=1, max_p=500):
    best_ratio = None
    best_pair = None
    for p in range(2, max_p+1):
        for q in range(p+1, max_p+1):
            D = p*q + p + q
            n_mod = (factor * pow(3, K, D) + c) % D
            if n_mod == 0:
                ratio = Fraction(p*q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p,q,D)
    return best_ratio, best_pair

for c in [265710644, 44636594]:
    r, pair = best_ratio_for_n_general(c, max_p=500)
    print(c, r, pair)


### Turn 55
def best_ratio_for_n_general(c, factor=1, max_p=500):
    best_ratio = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(p + 1, max_p + 1):
            D = p * q + p + q
            n_mod = (factor * pow(3, K, D) + c) % D
            if n_mod == 0:
                ratio = Fraction(p * q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p, q, D)
    return (best_ratio, best_pair)
for c in [265710644, 44636594]:
    r, pair = best_ratio_for_n_general(c, max_p=500)
    print(c, r, pair)

### Turn 56
max_D = 300000
moduli = list(range(1, max_D+1))
# But computing pow for each d up to 300k may be heavy but we can do incremental.

# We'll compute M_mod for needed D only (list of D from pairs)
Ds = set()
max_p = 500
for p in range(2, max_p+1):
    for q in range(p+1, max_p+1):
        D = p*q + p + q
        Ds.add(D)
len(Ds)


### Turn 56
max_D = 300000
moduli = list(range(1, max_D + 1))
Ds = set()
max_p = 500
for p in range(2, max_p + 1):
    for q in range(p + 1, max_p + 1):
        D = p * q + p + q
        Ds.add(D)
print(len(Ds))

### Turn 57
M_mods_big = {}
for D in Ds:
    M_mods_big[D] = pow(3, K, D)
len(M_mods_big)


### Turn 57
M_mods_big = {}
for D in Ds:
    M_mods_big[D] = pow(3, K, D)
print(len(M_mods_big))

### Turn 58
# Precompute M_mod for D up to 2000
max_D_small = 2000
M_mod_small = {d: pow(3, K, d) for d in range(2, max_D_small+1)}
def best_ratio_bruteforce(c, factor=1, p_limit=200):
    best_ratio = None
    best_pair = None
    for p in range(2, p_limit+1):
        for q in range(p+1, p_limit+1):
            D = p*q + p + q
            if D > max_D_small:
                # compute on the fly
                M_mod = pow(3, K, D)
            else:
                M_mod = M_mod_small[D]
            if (factor * M_mod + c) % D == 0:
                ratio = Fraction(p*q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p,q,D)
    return best_ratio, best_pair

for c in [265710644, 44636594]:
    r, pair = best_ratio_bruteforce(c, p_limit=200)
    print(c, r, pair)


### Turn 58
max_D_small = 2000
M_mod_small = {d: pow(3, K, d) for d in range(2, max_D_small + 1)}

def best_ratio_bruteforce(c, factor=1, p_limit=200):
    best_ratio = None
    best_pair = None
    for p in range(2, p_limit + 1):
        for q in range(p + 1, p_limit + 1):
            D = p * q + p + q
            if D > max_D_small:
                M_mod = pow(3, K, D)
            else:
                M_mod = M_mod_small[D]
            if (factor * M_mod + c) % D == 0:
                ratio = Fraction(p * q, D)
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (p, q, D)
    return (best_ratio, best_pair)
for c in [265710644, 44636594]:
    r, pair = best_ratio_bruteforce(c, p_limit=200)
    print(c, r, pair)

### Turn 59
for c in [265710644, 44636594]:
    r, pair = best_ratio_bruteforce(c, p_limit=100)
    print(c, r, pair)


### Turn 59
for c in [265710644, 44636594]:
    r, pair = best_ratio_bruteforce(c, p_limit=100)
    print(c, r, pair)

### Turn 60
def best_ratio_p2(c, factor=1, max_q=200):
    best = None
    best_q = None
    for q in range(3, max_q+1):
        D = 3*q + 2
        if (factor * pow(3, K, D) + c) % D == 0:
            ratio = Fraction(2*q, D)
            if best is None or ratio < best:
                best = ratio
                best_q = q
    return best, best_q

for c in [265710644, 44636594]:
    r,q = best_ratio_p2(c, max_q=200)
    print(c, r, q)


### Turn 60
def best_ratio_p2(c, factor=1, max_q=200):
    best = None
    best_q = None
    for q in range(3, max_q + 1):
        D = 3 * q + 2
        if (factor * pow(3, K, D) + c) % D == 0:
            ratio = Fraction(2 * q, D)
            if best is None or ratio < best:
                best = ratio
                best_q = q
    return (best, best_q)
for c in [265710644, 44636594]:
    r, q = best_ratio_p2(c, max_q=200)
    print(c, r, q)

### Turn 61
c=44636594
print(n_mod_d(c,19))


### Turn 61
c = 44636594
print(n_mod_d(c, 19))

### Turn 62
def best_ratio_all(c, max_p=200):
    best = None
    best_pair = None
    for p in range(2, max_p+1):
        for q in range(p+1, max_p+1):
            D = p*q + p + q
            if (pow(3, K, D) + c) % D == 0:
                ratio = Fraction(p*q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p,q,D)
    return best, best_pair

c=1848374
best, pair = best_ratio_all(c, max_p=200)
best, pair


### Turn 62
def best_ratio_all(c, max_p=200):
    best = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(p + 1, max_p + 1):
            D = p * q + p + q
            if (pow(3, K, D) + c) % D == 0:
                ratio = Fraction(p * q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p, q, D)
    return (best, best_pair)
c = 1848374
best, pair = best_ratio_all(c, max_p=200)
print((best, pair))

### Turn 63
def best_ratio_using_precomp(c, max_p=200):
    best = None
    best_pair = None
    for p in range(2, max_p+1):
        for q in range(p+1, max_p+1):
            D = p*q + p + q
            M_mod = M_mods[D]
            if (M_mod + c) % D == 0:
                ratio = Fraction(p*q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p,q,D)
    return best, best_pair

c = 1848374
best, pair = best_ratio_using_precomp(c, max_p=200)
best, pair


### Turn 63
def best_ratio_using_precomp(c, max_p=200):
    best = None
    best_pair = None
    for p in range(2, max_p + 1):
        for q in range(p + 1, max_p + 1):
            D = p * q + p + q
            M_mod = M_mods[D]
            if (M_mod + c) % D == 0:
                ratio = Fraction(p * q, D)
                if best is None or ratio < best:
                    best = ratio
                    best_pair = (p, q, D)
    return (best, best_pair)
c = 1848374
best, pair = best_ratio_using_precomp(c, max_p=200)
print((best, pair))

### Turn 64
c=10162574
print(n_mod_d(c,19))


### Turn 64
c = 10162574
print(n_mod_d(c, 19))

### Turn 65
print(n_mod_d(c,14))


### Turn 65
print(n_mod_d(c, 14))

### Turn 66
print(n_mod_d(c,17))


### Turn 66
print(n_mod_d(c, 17))

### Turn 67
c=44636594
print("mod 23:", n_mod_d(c,23))


### Turn 67
c = 44636594
print('mod 23:', n_mod_d(c, 23))

### Turn 68
print("mod 29:", n_mod_d(c,29))


### Turn 68
print('mod 29:', n_mod_d(c, 29))

### Turn 69
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
p,q = S.numerator, S.denominator
(p+q) % 99991


### Turn 69
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
p, q = (S.numerator, S.denominator)
print((p + q) % 99991)

### Turn 70
(p+q) % 99991


### Turn 70
print((p + q) % 99991)

### Turn 71
c=10162574
print("n mod 47:", n_mod_d(c,47))


### Turn 71
c = 10162574
print('n mod 47:', n_mod_d(c, 47))

### Turn 72
c=1848374
print(n_mod_d(c,53))


### Turn 72
c = 1848374
print(n_mod_d(c, 53))

### Turn 73
c=265710644
print(n_mod_d(c,97))


### Turn 73
c = 265710644
print(n_mod_d(c, 97))

### Turn 74
c=44636594
print(n_mod_d(c,167))


### Turn 74
c = 44636594
print(n_mod_d(c, 167))

### Turn 75
from fractions import Fraction
terms = [Fraction(2,3), Fraction(10,3), Fraction(16,25), Fraction(30,47), Fraction(64,97), Fraction(110,167)]
S = sum(terms, Fraction(0,1))
S


### Turn 75
from fractions import Fraction
terms = [Fraction(2, 3), Fraction(10, 3), Fraction(16, 25), Fraction(30, 47), Fraction(64, 97), Fraction(110, 167)]
S = sum(terms, Fraction(0, 1))
print(S)

### Turn 76
(p+q) % 99991


### Turn 76
print((p + q) % 99991)

### Turn 77
c=265710644
print("mod 41:", n_mod_d(c,41))


### Turn 77
c = 265710644
print('mod 41:', n_mod_d(c, 41))

### Turn 78
print("mod 23:", n_mod_d(c,23))


### Turn 78
print('mod 23:', n_mod_d(c, 23))

### Turn 79
c = 0  # for n=M
# n = 5M => factor 5
def n_mod_5M(d):
    # n = 5M, c=0 but factor 5
    return (5 * pow(3, K, d)) % d

print("mod 11:", n_mod_5M(11))


### Turn 79
c = 0

def n_mod_5M(d):
    return 5 * pow(3, K, d) % d
print('mod 11:', n_mod_5M(11))

### Turn 80
def n_mod_5M(d):
    return (5 * pow(3, K, d)) % d
for d in [14,17,23,26,29,31]:
    print(d, n_mod_5M(d))


### Turn 80
def n_mod_5M(d):
    return 5 * pow(3, K, d) % d
for d in [14, 17, 23, 26, 29, 31]:
    print(d, n_mod_5M(d))

