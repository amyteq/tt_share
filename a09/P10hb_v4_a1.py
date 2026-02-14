# Auto-extracted from: P10hb_v4.md
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



### Turn 0
import math, sympy as sp, itertools, sys, fractions, decimal, collections, random, math, itertools, functools, string, json, re, hashlib, os, pprint, textwrap, fractions, typing, numbers, statistics, fractions, itertools, math, sympy as sp
def exponent_in_factorial(n, p):
    e = 0
    power = p
    while power <= n:
        e += n // power
        power *= p
    return e
exponent_in_factorial(2025, 3)


### Turn 0
import math, sympy as sp, itertools, sys, fractions, decimal, collections, random, math, itertools, functools, string, json, re, hashlib, os, pprint, textwrap, fractions, typing, numbers, statistics, fractions, itertools, math, sympy as sp

def exponent_in_factorial(n, p):
    e = 0
    power = p
    while power <= n:
        e += n // power
        power *= p
    return e
print(exponent_in_factorial(2025, 3))

### Turn 1
def f_brutal(N):
    min_l = None
    # iterate triples a<b<c? distinct unordered
    # just iterate a from 1..N-2; b from a+1..N-a-1; c=N-a-b, ensure c>b
    for a in range(1, N-1):
        for b in range(a+1, N-a):
            c = N - a - b
            if c <= b:
                continue
            # distinct positive
            import math
            l = math.lcm(a,b,c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def compute_f_list(limit):
    res={}
    for N in range(6, limit+1):
        res[N]=f_brutal(N)
    return res

fvals=compute_f_list(30)
fvals


### Turn 1
def f_brutal(N):
    min_l = None
    for a in range(1, N - 1):
        for b in range(a + 1, N - a):
            c = N - a - b
            if c <= b:
                continue
            import math
            l = math.lcm(a, b, c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def compute_f_list(limit):
    res = {}
    for N in range(6, limit + 1):
        res[N] = f_brutal(N)
    return res
fvals = compute_f_list(30)
print(fvals)

### Turn 2
def compute_f_upto(Nmax):
    return [f_brutal(N) for N in range(Nmax+1)]

list_f = compute_f_upto(60)
list_f[:61]


### Turn 2
def compute_f_upto(Nmax):
    return [f_brutal(N) for N in range(Nmax + 1)]
list_f = compute_f_upto(60)
print(list_f[:61])

### Turn 3
data = [(N, f_brutal(N)) for N in range(6, 101)]
data[:30]


### Turn 3
data = [(N, f_brutal(N)) for N in range(6, 101)]
print(data[:30])

### Turn 4
data[30:60]


### Turn 4
print(data[30:60])

### Turn 5
data[60:90]


### Turn 5
print(data[60:90])

### Turn 6
data[90:120]


### Turn 6
print(data[90:120])

### Turn 7
ratios = [(N, f_brutal(N)/N) for N in range(20,101,5)]
ratios[:10]


### Turn 7
ratios = [(N, f_brutal(N) / N) for N in range(20, 101, 5)]
print(ratios[:10])

### Turn 8
def f_fast(N):
    # naive O(N^2), but fine for N up to 500
    min_l = N # start with N maybe
    for a in range(1, N-1):
        for b in range(a+1, N-a):
            c = N - a - b
            if c <= b:
                continue
            l = math.lcm(a,b,c)
            if l < min_l:
                min_l = l
                # early break? Not necessary
    return min_l

vals = [f_fast(N) for N in range(6, 101)]
vals[:20]


### Turn 8
def f_fast(N):
    min_l = N
    for a in range(1, N - 1):
        for b in range(a + 1, N - a):
            c = N - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if l < min_l:
                min_l = l
    return min_l
vals = [f_fast(N) for N in range(6, 101)]
print(vals[:20])

### Turn 9
import math, itertools, sys, time, collections, math, random, functools, itertools, math
def compute_f_upto(Nmax):
    f = [0]*(Nmax+1)
    for N in range(6, Nmax+1):
        min_l = None
        # to speed up, we could iterate a up to N//3 etc.
        for a in range(1, N//3+1):
            for b in range(a+1, (N-a)//2+1):
                c = N - a - b
                if c <= b:
                    continue
                l = math.lcm(a,b,c)
                if min_l is None or l < min_l:
                    min_l = l
        f[N] = min_l
    return f

# compute up to 200
fvals200 = compute_f_upto(200)
list(zip(range(6,51), fvals200[6:51]))[:45]


### Turn 9
import math, itertools, sys, time, collections, math, random, functools, itertools, math

def compute_f_upto(Nmax):
    f = [0] * (Nmax + 1)
    for N in range(6, Nmax + 1):
        min_l = None
        for a in range(1, N // 3 + 1):
            for b in range(a + 1, (N - a) // 2 + 1):
                c = N - a - b
                if c <= b:
                    continue
                l = math.lcm(a, b, c)
                if min_l is None or l < min_l:
                    min_l = l
        f[N] = min_l
    return f
fvals200 = compute_f_upto(200)
print(list(zip(range(6, 51), fvals200[6:51]))[:45])

### Turn 10
ratios_large = [(N, fvals200[N]/N) for N in range(6, 201)]
max_ratio = max(ratios_large, key=lambda x: x[1])
min_ratio = min(ratios_large, key=lambda x: x[1])
max_ratio, min_ratio


### Turn 10
ratios_large = [(N, fvals200[N] / N) for N in range(6, 201)]
max_ratio = max(ratios_large, key=lambda x: x[1])
min_ratio = min(ratios_large, key=lambda x: x[1])
print((max_ratio, min_ratio))

### Turn 11
def candidate1(N):
    n1 = N-1
    # factor n1 to find smallest odd prime divisor
    for p in sp.factorint(n1):
        # we need smallest odd prime divisor >2
        pass


### Turn 11
def candidate1(N):
    n1 = N - 1
    for p in sp.factorint(n1):
        pass

### Turn 12
c_values = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_values]
mods


### Turn 12
c_values = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_values]
print(mods)

### Turn 13
c_mod3 = [c % 3 for c in c_values]
c_mod3


### Turn 13
c_mod3 = [c % 3 for c in c_values]
print(c_mod3)

### Turn 14
import math, sympy as sp, itertools, sys, functools, random, collections, math, fractions, itertools, math, time, sys, decimal, itertools, math

### Turn 14
import math, sympy as sp, itertools, sys, functools, random, collections, math, fractions, itertools, math, time, sys, decimal, itertools, math

### Turn 15
import math, time, sys, itertools, functools, collections, math, sympy as sp
start = time.time()
A = math.factorial(2025)
elapsed = time.time() - start
print("Factorial computed, digits:", len(str(A)), "time", elapsed)


### Turn 15
import math, time, sys, itertools, functools, collections, math, sympy as sp
start = time.time()
A = math.factorial(2025)
elapsed = time.time() - start
print('Factorial computed, digits:', len(str(A)), 'time', elapsed)

### Turn 16
import sys, math, time
sys.set_int_max_str_digits(10000)
A = math.factorial(2025)
print("digits:", len(str(A)))


### Turn 16
import sys, math, time
sys.set_int_max_str_digits(10000)
A = math.factorial(2025)
print('digits:', len(str(A)))

### Turn 17
pairs = []
max_val = 30
for p in range(2, max_val+1):
    for q in range(p+1, max_val+1):
        D = p*q + p + q
        pairs.append((p,q,D))
len(pairs)


### Turn 17
pairs = []
max_val = 30
for p in range(2, max_val + 1):
    for q in range(p + 1, max_val + 1):
        D = p * q + p + q
        pairs.append((p, q, D))
print(len(pairs))

### Turn 18
import sympy as sp, math, itertools, sys, time
# compute M_mod_d for needed d values (D from pairs)
def compute_M_mods(moduli):
    M_mod = {}
    for d in moduli:
        M_mod[d] = pow(3, A, d)
    return M_mod

# get unique D values from pairs
D_values = sorted(set(D for (_,_,D) in pairs))
len(D_values)


### Turn 18
import sympy as sp, math, itertools, sys, time

def compute_M_mods(moduli):
    M_mod = {}
    for d in moduli:
        M_mod[d] = pow(3, A, d)
    return M_mod
D_values = sorted(set((D for _, _, D in pairs)))
print(len(D_values))

### Turn 19
M_mod_D = compute_M_mods(D_values)
print("Computed M_mod for", len(M_mod_D), "moduli")


### Turn 19
M_mod_D = compute_M_mods(D_values)
print('Computed M_mod for', len(M_mod_D), 'moduli')

### Turn 20
odd_primes = list(sp.primerange(3, 2000))
len(odd_primes)


### Turn 20
odd_primes = list(sp.primerange(3, 2000))
print(len(odd_primes))

### Turn 21
# Define cases as tuples: (name, kind, offset)
# kind: 'M', '5M', 'Mplus'
cases = [
    ('case0', 'M', 0),  # N = M
    ('case5M', '5M', 0),  # N = 5M
    ('case1', 'Mplus', 1848374),
    ('case2', 'Mplus', 10162574),
    ('case3', 'Mplus', 265710644),
    ('case4', 'Mplus', 44636594)
]


### Turn 21
cases = [('case0', 'M', 0), ('case5M', '5M', 0), ('case1', 'Mplus', 1848374), ('case2', 'Mplus', 10162574), ('case3', 'Mplus', 265710644), ('case4', 'Mplus', 44636594)]
print(cases)

### Turn 22
from fractions import Fraction

def minimal_alpha_for_case(kind, offset):
    # factor
    factor = 5 if kind == '5M' else 1
    # compute minimal alpha_total
    min_alpha = Fraction(10**9, 1)  # large placeholder
    # candidate from p,q pairs
    for (p,q,D) in pairs:
        # compute N_mod_D
        if kind == 'M':
            N_mod = M_mod_D[D]
        elif kind == '5M':
            N_mod = (5 * M_mod_D[D]) % D
        else:  # 'Mplus'
            N_mod = (M_mod_D[D] + offset % D) % D
        if N_mod == 0:
            alpha_N = Fraction(p*q, D)
            alpha_total = alpha_N * factor
            if alpha_total < min_alpha:
                min_alpha = alpha_total
    # candidate from divisor method
    # need smallest odd prime divisor r of N-1
    # compute N-1 mod p for p up to maybe 2000
    r = None
    for prime in odd_primes:
        if kind == 'M':
            mod_val = (M_mod_D.get(prime, pow(3, A, prime)) - 1) % prime
        elif kind == '5M':
            mod_val = (5 * pow(3, A, prime) - 1) % prime
        else:
            mod_val = (M_mod_D.get(prime, pow(3, A, prime)) + (offset-1) % prime) % prime
        if mod_val == 0:
            r = prime
            break
    if r is None:
        # maybe N-1 is prime >2000; fallback to using r = N-1? But then α≈(r-1)/r near 1, not minimal.
        # We'll treat as no candidate from this method.
        pass
    else:
        alpha_N = Fraction(r-1, r)
        alpha_total = alpha_N * factor
        if alpha_total < min_alpha:
            min_alpha = alpha_total
    return min_alpha, r

# compute minimal alphas for each case
results = {}
for name, kind, offset in cases:
    alpha, r = minimal_alpha_for_case(kind, offset)
    results[name] = (alpha, r)
    print(name, "alpha:", alpha, "r:", r)


### Turn 22
from fractions import Fraction

def minimal_alpha_for_case(kind, offset):
    factor = 5 if kind == '5M' else 1
    min_alpha = Fraction(10 ** 9, 1)
    for p, q, D in pairs:
        if kind == 'M':
            N_mod = M_mod_D[D]
        elif kind == '5M':
            N_mod = 5 * M_mod_D[D] % D
        else:
            N_mod = (M_mod_D[D] + offset % D) % D
        if N_mod == 0:
            alpha_N = Fraction(p * q, D)
            alpha_total = alpha_N * factor
            if alpha_total < min_alpha:
                min_alpha = alpha_total
    r = None
    for prime in odd_primes:
        if kind == 'M':
            mod_val = (M_mod_D.get(prime, pow(3, A, prime)) - 1) % prime
        elif kind == '5M':
            mod_val = (5 * pow(3, A, prime) - 1) % prime
        else:
            mod_val = (M_mod_D.get(prime, pow(3, A, prime)) + (offset - 1) % prime) % prime
        if mod_val == 0:
            r = prime
            break
    if r is None:
        pass
    else:
        alpha_N = Fraction(r - 1, r)
        alpha_total = alpha_N * factor
        if alpha_total < min_alpha:
            min_alpha = alpha_total
    return (min_alpha, r)
results = {}
for name, kind, offset in cases:
    alpha, r = minimal_alpha_for_case(kind, offset)
    results[name] = (alpha, r)
    print(name, 'alpha:', alpha, 'r:', r)

### Turn 23
def find_pair_for_alpha(target):
    for (p,q,D) in pairs:
        if Fraction(p*q, D) == target:
            return (p,q,D)
    return None

print(find_pair_for_alpha(Fraction(2,3)))


### Turn 23
def find_pair_for_alpha(target):
    for p, q, D in pairs:
        if Fraction(p * q, D) == target:
            return (p, q, D)
    return None
print(find_pair_for_alpha(Fraction(2, 3)))

### Turn 24
print(find_pair_for_alpha(Fraction(34,53)))


### Turn 24
print(find_pair_for_alpha(Fraction(34, 53)))

### Turn 25
print(find_pair_for_alpha(Fraction(30,47)))


### Turn 25
print(find_pair_for_alpha(Fraction(30, 47)))

### Turn 26
print(find_pair_for_alpha(Fraction(75,103)))


### Turn 26
print(find_pair_for_alpha(Fraction(75, 103)))

### Turn 27
# compute T = sum floor(A * num / den)
terms = [
    (2, 3), # case0
    (10, 3), # case5M
    (34, 53), # case1
    (30, 47), # case2
    (75, 103), # case3
    (75, 103), # case4
]
T = 0
for num, den in terms:
    term = (A * num) // den
    T += term
print("T computed")


### Turn 27
terms = [(2, 3), (10, 3), (34, 53), (30, 47), (75, 103), (75, 103)]
T = 0
for num, den in terms:
    term = A * num // den
    T += term
print('T computed')

### Turn 28
g = math.gcd(T, A)
p = T // g
q = A // g
print("gcd:", g)
print("p digits:", len(str(p)))
print("q digits:", len(str(q)))


### Turn 28
g = math.gcd(T, A)
p = T // g
q = A // g
print('gcd:', g)
print('p digits:', len(str(p)))
print('q digits:', len(str(q)))

### Turn 29
print("p:", p)
print("q:", q)


### Turn 29
print('p:', p)
print('q:', q)

### Turn 30
mod = (p + q) % 99991
mod


### Turn 30
mod = (p + q) % 99991
print(mod)

### Turn 31
def f_algorithm(N):
    # find minimal LCM via method
    # first compute minimal via pair method
    min_L = None
    # we need to get candidate L = N * p*q / D, must be integer.
    # We'll generate pairs up to p,q <= 30 for now.
    for (p,q,D) in pairs:
        if N % D == 0:
            L = N * p * q // D
            if min_L is None or L < min_L:
                min_L = L
    # divisor method
    # find smallest odd prime divisor r of N-1
    r_small = None
    for prime in odd_primes:
        if (N-1) % prime == 0:
            r_small = prime
            break
    if r_small is not None:
        L2 = (N-1) * (r_small - 1) // r_small
        if min_L is None or L2 < min_L:
            min_L = L2
    return min_L

# test for N up to 200
ok = True
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm(N)
    if brute != algo:
        print("Mismatch at", N, brute, algo)
        ok = False
        break
print("All matched up to 200:", ok)


### Turn 31
def f_algorithm(N):
    min_L = None
    for p, q, D in pairs:
        if N % D == 0:
            L = N * p * q // D
            if min_L is None or L < min_L:
                min_L = L
    r_small = None
    for prime in odd_primes:
        if (N - 1) % prime == 0:
            r_small = prime
            break
    if r_small is not None:
        L2 = (N - 1) * (r_small - 1) // r_small
        if min_L is None or L2 < min_L:
            min_L = L2
    return min_L
ok = True
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm(N)
    if brute != algo:
        print('Mismatch at', N, brute, algo)
        ok = False
        break
print('All matched up to 200:', ok)

### Turn 32
def f_algorithm(N):
    # minimal L
    min_L = None
    # pair method
    for (p,q,D) in pairs:
        if N % D == 0:
            L = N * p * q // D
            if min_L is None or L < min_L:
                min_L = L
    # divisor method: smallest odd prime divisor p of N-1 with p < N-1
    r_small = None
    for prime in odd_primes:
        if (N-1) % prime == 0 and (N-1)//prime > 1:
            r_small = prime
            break
    if r_small is not None:
        L2 = (N-1) * (r_small - 1) // r_small
        if min_L is None or L2 < min_L:
            min_L = L2
    return min_L

ok=True
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm(N)
    if brute != algo:
        print("Mismatch at", N, brute, algo)
        ok=False
        break
print("All matched up to 200:", ok)


### Turn 32
def f_algorithm(N):
    min_L = None
    for p, q, D in pairs:
        if N % D == 0:
            L = N * p * q // D
            if min_L is None or L < min_L:
                min_L = L
    r_small = None
    for prime in odd_primes:
        if (N - 1) % prime == 0 and (N - 1) // prime > 1:
            r_small = prime
            break
    if r_small is not None:
        L2 = (N - 1) * (r_small - 1) // r_small
        if min_L is None or L2 < min_L:
            min_L = L2
    return min_L
ok = True
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm(N)
    if brute != algo:
        print('Mismatch at', N, brute, algo)
        ok = False
        break
print('All matched up to 200:', ok)

### Turn 33
def f_and_structure(N):
    # return minimal LCM and whether optimal includes L in the triple (i.e., t in divisors)
    min_l = None
    includes_t = False
    best_trip = None
    for a in range(1, N-1):
        for b in range(a+1, N-a):
            c = N - a - b
            if c <= b:
                continue
            t = math.lcm(a,b,c)
            if min_l is None or t < min_l:
                min_l = t
                # check if any of a,b,c equals t
                includes = (a == t or b == t or c == t)
                includes_t = includes
                best_trip = (a,b,c)
    return min_l, includes_t, best_trip

# test some N
for N in range(6, 31):
    t, inc, trip = f_and_structure(N)
    if inc:
        inc_str = "inc"
    else:
        inc_str = "no"
    # print
    if N <= 20:
        print(N, t, inc_str, trip)


### Turn 33
def f_and_structure(N):
    min_l = None
    includes_t = False
    best_trip = None
    for a in range(1, N - 1):
        for b in range(a + 1, N - a):
            c = N - a - b
            if c <= b:
                continue
            t = math.lcm(a, b, c)
            if min_l is None or t < min_l:
                min_l = t
                includes = a == t or b == t or c == t
                includes_t = includes
                best_trip = (a, b, c)
    return (min_l, includes_t, best_trip)
for N in range(6, 31):
    t, inc, trip = f_and_structure(N)
    if inc:
        inc_str = 'inc'
    else:
        inc_str = 'no'
    if N <= 20:
        print(N, t, inc_str, trip)

### Turn 34
def test_small(k):
    M = 3**k
    t = 2 * 3**(k-1) # (2/3)M
    # find three distinct divisors of t summing to M
    divisors = sp.divisors(t)
    # need three distinct divisors
    for a in divisors:
        for b in divisors:
            if b <= a: continue
            c = M - a - b
            if c <= b: continue
            if c in divisors and c != a and c != b:
                return (a,b,c)
    return None

for k in range(2, 10):
    sol = test_small(k)
    print(k, sol)


### Turn 34
def test_small(k):
    M = 3 ** k
    t = 2 * 3 ** (k - 1)
    divisors = sp.divisors(t)
    for a in divisors:
        for b in divisors:
            if b <= a:
                continue
            c = M - a - b
            if c <= b:
                continue
            if c in divisors and c != a and (c != b):
                return (a, b, c)
    return None
for k in range(2, 10):
    sol = test_small(k)
    print(k, sol)

### Turn 35
def test_range(limit):
    mismatches = []
    for N in range(6, limit+1):
        brute = f_brutal(N)
        algo = f_algorithm(N)
        # handle case where algo returns None (if no candidate)
        if algo is None or brute != algo:
            mismatches.append((N, brute, algo))
    return mismatches

mism = test_range(200)
mism[:20], len(mism)


### Turn 35
def test_range(limit):
    mismatches = []
    for N in range(6, limit + 1):
        brute = f_brutal(N)
        algo = f_algorithm(N)
        if algo is None or brute != algo:
            mismatches.append((N, brute, algo))
    return mismatches
mism = test_range(200)
print((mism[:20], len(mism)))

### Turn 36
def f_algorithm_enhanced(N, max_pq=30):
    # compute all candidate L values
    min_L = None
    # candidate from t = N if possible
    # find three distinct divisors of N summing to N
    divisors = sp.divisors(N)
    # check combos
    for i in range(len(divisors)):
        a = divisors[i]
        for j in range(i+1, len(divisors)):
            b = divisors[j]
            c = N - a - b
            if c <= b or c == a or c == b:
                continue
            if c in divisors and c != a and c != b:
                # triple found
                if min_L is None or N < min_L:
                    min_L = N
    # pair method (t = N * p*q / D)
    for p in range(2, max_pq+1):
        for q in range(p+1, max_pq+1):
            D = p*q + p + q
            if N % D == 0:
                L = N * p * q // D
                if min_L is None or L < min_L:
                    min_L = L
    # divisor method
    # find smallest odd prime divisor r of N-1 such that (N-1)/r > 1
    for prime in odd_primes:
        if (N-1) % prime == 0 and (N-1)//prime > 1:
            L2 = (N-1) * (prime - 1) // prime
            if min_L is None or L2 < min_L:
                min_L = L2
            break
    # also consider using divisor d not maximal? maybe we need to try larger d? But we'll ignore for now
    return min_L

# test up to 200
mismatches = []
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm_enhanced(N, max_pq=30)
    if brute != algo:
        mismatches.append((N, brute, algo))
len(mismatches), mismatches[:10]


### Turn 36
def f_algorithm_enhanced(N, max_pq=30):
    min_L = None
    divisors = sp.divisors(N)
    for i in range(len(divisors)):
        a = divisors[i]
        for j in range(i + 1, len(divisors)):
            b = divisors[j]
            c = N - a - b
            if c <= b or c == a or c == b:
                continue
            if c in divisors and c != a and (c != b):
                if min_L is None or N < min_L:
                    min_L = N
    for p in range(2, max_pq + 1):
        for q in range(p + 1, max_pq + 1):
            D = p * q + p + q
            if N % D == 0:
                L = N * p * q // D
                if min_L is None or L < min_L:
                    min_L = L
    for prime in odd_primes:
        if (N - 1) % prime == 0 and (N - 1) // prime > 1:
            L2 = (N - 1) * (prime - 1) // prime
            if min_L is None or L2 < min_L:
                min_L = L2
            break
    return min_L
mismatches = []
for N in range(6, 201):
    brute = f_brutal(N)
    algo = f_algorithm_enhanced(N, max_pq=30)
    if brute != algo:
        mismatches.append((N, brute, algo))
print((len(mismatches), mismatches[:10]))

### Turn 37
def f_brutal_limit(N):
    min_l = None
    for a in range(1, N-1):
        for b in range(a+1, N-a):
            c = N - a - b
            if c <= b:
                continue
            l = math.lcm(a,b,c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_candidate(N):
    # use pair method to find minimal L = N * p*q / D where D divides N
    min_L = None
    # pair method with p,q up to 30
    for p in range(2, 30):
        for q in range(p+1, 30):
            D = p*q + p + q
            if N % D == 0:
                L = N * p * q // D
                if min_L is None or L < min_L:
                    min_L = L
    # divisor method (1,d)
    # smallest odd prime divisor r of N-1, r>2 and (N-1)/r > 1
    for prime in odd_primes:
        if (N-1) % prime == 0 and (N-1)//prime > 1:
            L2 = (N-1) * (prime - 1) // prime
            if min_L is None or L2 < min_L:
                min_L = L2
            break
    # also consider t=N if N can be expressed as sum of three distinct divisors of N
    # quick check: if there exist a,b distinct divisors of N with a+b = N - c where c is another divisor distinct.
    # but this is complex; ignore.
    return min_L

# test for M = 3^7
k = 7
M_small = 3**k
test_offsets = [0, 1, 2, 3, 5, 10, 100, 500, 1000]
for offset in test_offsets:
    N = M_small + offset
    brute = f_brutal_limit(N)
    cand = f_candidate(N)
    print("N=M+%d: brute=%d cand=%s" % (offset, brute, cand))


### Turn 37
def f_brutal_limit(N):
    min_l = None
    for a in range(1, N - 1):
        for b in range(a + 1, N - a):
            c = N - a - b
            if c <= b:
                continue
            l = math.lcm(a, b, c)
            if min_l is None or l < min_l:
                min_l = l
    return min_l

def f_candidate(N):
    min_L = None
    for p in range(2, 30):
        for q in range(p + 1, 30):
            D = p * q + p + q
            if N % D == 0:
                L = N * p * q // D
                if min_L is None or L < min_L:
                    min_L = L
    for prime in odd_primes:
        if (N - 1) % prime == 0 and (N - 1) // prime > 1:
            L2 = (N - 1) * (prime - 1) // prime
            if min_L is None or L2 < min_L:
                min_L = L2
            break
    return min_L
k = 7
M_small = 3 ** k
test_offsets = [0, 1, 2, 3, 5, 10, 100, 500, 1000]
for offset in test_offsets:
    N = M_small + offset
    brute = f_brutal_limit(N)
    cand = f_candidate(N)
    print('N=M+%d: brute=%d cand=%s' % (offset, brute, cand))

### Turn 38
M_small % 53


### Turn 38
print(M_small % 53)

### Turn 39
offset = 39
N = M_small + offset
brute = f_brutal_limit(N)
cand = f_candidate(N)
brute, cand


### Turn 39
offset = 39
N = M_small + offset
brute = f_brutal_limit(N)
cand = f_candidate(N)
print((brute, cand))

### Turn 40
p+q


### Turn 40
print(p + q)

### Turn 41
# recompute to verify
from math import gcd
# re-calc T
terms = [
    (2, 3), # case0
    (10, 3), # case5M
    (34, 53),
    (30, 47),
    (75, 103),
    (75, 103)
]
T2 = sum((A * num) // den for num, den in terms)
g = gcd(T2, A)
p2 = T2 // g
q2 = A // g
print(p2, q2)
print(g)
print((p2+q2) % 99991)


### Turn 41
from math import gcd
terms = [(2, 3), (10, 3), (34, 53), (30, 47), (75, 103), (75, 103)]
T2 = sum((A * num // den for num, den in terms))
g = gcd(T2, A)
p2 = T2 // g
q2 = A // g
print(p2, q2)
print(g)
print((p2 + q2) % 99991)

