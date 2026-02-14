# Auto-extracted from: P10hb_v4.md
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



### Turn 0
import sympy as sp

def f_bruteforce(N, limit=1000):
    # find smallest X
    # iterate X from 1 upwards (maybe up to N)
    for X in range(1, N+1):
        # list of distinct divisors of X
        divs = sp.divisors(X)
        # need three distinct divisors
        if len(divs) < 3:
            continue
        # check all triples
        for i in range(len(divs)):
            for j in range(i+1, len(divs)):
                for k in range(j+1, len(divs)):
                    if divs[i] + divs[j] + divs[k] == N:
                        return X
    return None

def f_fast(N, maxX=None):
    # just use brute for now
    return f_bruteforce(N)
    
# compute f(N) for N up to 100 maybe
f_values = {}
for N in range(6, 101):
    f_values[N] = f_bruteforce(N)
f_values[:10]


### Turn 0
import sympy as sp

def f_bruteforce(N, limit=1000):
    for X in range(1, N + 1):
        divs = sp.divisors(X)
        if len(divs) < 3:
            continue
        for i in range(len(divs)):
            for j in range(i + 1, len(divs)):
                for k in range(j + 1, len(divs)):
                    if divs[i] + divs[j] + divs[k] == N:
                        return X
    return None

def f_fast(N, maxX=None):
    return f_bruteforce(N)
f_values = {}
for N in range(6, 101):
    f_values[N] = f_bruteforce(N)
print(f_values[:10])

### Turn 1
for N in range(6, 31):
    print(N, f_values[N])


### Turn 1
for N in range(6, 31):
    print(N, f_values[N])

### Turn 2
def f_bruteforce(N, maxX=5000):
    for X in range(1, maxX+1):
        divs = sp.divisors(X)
        if len(divs) < 3:
            continue
        # choose any three distinct
        for combo in sp.utilities.iterables.combinations(divs, 3):
            if sum(combo) == N:
                return X
    return None

for N in range(6, 21):
    print(N, f_bruteforce(N, maxX=200))


### Turn 2
def f_bruteforce(N, maxX=5000):
    for X in range(1, maxX + 1):
        divs = sp.divisors(X)
        if len(divs) < 3:
            continue
        for combo in sp.utilities.iterables.combinations(divs, 3):
            if sum(combo) == N:
                return X
    return None
for N in range(6, 21):
    print(N, f_bruteforce(N, maxX=200))

### Turn 3
def compute_f_range(Nmax):
    res = {}
    for N in range(6, Nmax+1):
        res[N] = f_bruteforce(N, maxX=2000)
    return res

res = compute_f_range(200)
# print few
for N in range(6, 51):
    print(N, res[N])


### Turn 3
def compute_f_range(Nmax):
    res = {}
    for N in range(6, Nmax + 1):
        res[N] = f_bruteforce(N, maxX=2000)
    return res
res = compute_f_range(200)
for N in range(6, 51):
    print(N, res[N])

### Turn 4
ratios = [(N, res[N]/N) for N in range(6, 101)]
ratios[:10]


### Turn 4
ratios = [(N, res[N] / N) for N in range(6, 101)]
print(ratios[:10])

### Turn 5
def order_mod(a,m):
    for k in range(1,m):
        if pow(a,k,m)==1:
            return k
    return None

order_mod(3,11)


### Turn 5
def order_mod(a, m):
    for k in range(1, m):
        if pow(a, k, m) == 1:
            return k
    return None
print(order_mod(3, 11))

### Turn 6
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c%11 for c in c_vals]
mods


### Turn 6
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
mods = [c % 11 for c in c_vals]
print(mods)

### Turn 7
order_mod(3,7)


### Turn 7
print(order_mod(3, 7))

### Turn 8
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
mods7 = [c%7 for c in c_vals]
mods7


### Turn 8
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
mods7 = [c % 7 for c in c_vals]
print(mods7)

### Turn 9
mod19 = pow(3, sp.factorial(2025) % 18, 19)  # Actually order? Let's compute exponent modulo phi(19)=18 maybe.



### Turn 9
mod19 = pow(3, sp.factorial(2025) % 18, 19)
print(mod19)

### Turn 10
order_mod(3,19)


### Turn 10
print(order_mod(3, 19))

### Turn 11
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
[ (c % 19) for c in c_vals]


### Turn 11
c_vals = [0, 1848374, 10162574, 265710644, 44636594]
print([c % 19 for c in c_vals])

### Turn 12
order_mod(3,17)


### Turn 12
print(order_mod(3, 17))

### Turn 13
[c % 17 for c in c_vals]


### Turn 13
print([c % 17 for c in c_vals])

### Turn 14
order_mod(3,5)


### Turn 14
print(order_mod(3, 5))

### Turn 15
[c % 5 for c in c_vals]


### Turn 15
print([c % 5 for c in c_vals])

### Turn 16
def ratio(p,q):
    return (p*q)/(p*q + p + q)
best = 1
best_pair = None
for p in range(2,21):
    for q in range(p+1,21):
        r = ratio(p,q)
        if r < best:
            best = r
            best_pair = (p,q)
best, best_pair


### Turn 16
def ratio(p, q):
    return p * q / (p * q + p + q)
best = 1
best_pair = None
for p in range(2, 21):
    for q in range(p + 1, 21):
        r = ratio(p, q)
        if r < best:
            best = r
            best_pair = (p, q)
print((best, best_pair))

### Turn 17
def smallest_odd_prime_factor_of_3_pow_minus_one(k, limit=1000):
    n = pow(3,k)
    n_minus = n - 1
    # factor small primes up to limit
    for p in sp.primerange(3, limit):
        if n_minus % p == 0:
            return p
    return None

for k in range(1,21):
    spf = smallest_odd_prime_factor_of_3_pow_minus_one(k,200)
    print(k, spf)


### Turn 17
def smallest_odd_prime_factor_of_3_pow_minus_one(k, limit=1000):
    n = pow(3, k)
    n_minus = n - 1
    for p in sp.primerange(3, limit):
        if n_minus % p == 0:
            return p
    return None
for k in range(1, 21):
    spf = smallest_odd_prime_factor_of_3_pow_minus_one(k, 200)
    print(k, spf)

### Turn 18
order_mod(3,23)


### Turn 18
print(order_mod(3, 23))

### Turn 19
def D_val(p,q):
    D = p*q + p + q
    g = sp.gcd(p*q, D)
    return D // g

powers_of_3 = [3**k for k in range(1,10)]  # up to 3^9
pairs = []
for p in range(2,21):
    for q in range(p+1,21):
        D = D_val(p,q)
        if D in powers_of_3:
            pairs.append((p,q,D))
pairs


### Turn 19
def D_val(p, q):
    D = p * q + p + q
    g = sp.gcd(p * q, D)
    return D // g
powers_of_3 = [3 ** k for k in range(1, 10)]
pairs = []
for p in range(2, 21):
    for q in range(p + 1, 21):
        D = D_val(p, q)
        if D in powers_of_3:
            pairs.append((p, q, D))
print(pairs)

### Turn 20
pairs = []
powers_of_3_set = set([3**k for k in range(0,15)])  # include 1?
for p in range(2,101):
    for q in range(p+1,101):
        D = D_val(p,q)
        if D in powers_of_3_set:
            pairs.append((p,q,D))
pairs[:20], len(pairs)


### Turn 20
pairs = []
powers_of_3_set = set([3 ** k for k in range(0, 15)])
for p in range(2, 101):
    for q in range(p + 1, 101):
        D = D_val(p, q)
        if D in powers_of_3_set:
            pairs.append((p, q, D))
print((pairs[:20], len(pairs)))

### Turn 21
def ratio_pair(p,q):
    return sp.Rational(p*q, p*q + p + q)

pairs_info = [(p,q,D, ratio_pair(p,q)) for p,q,D in pairs]
pairs_info[:10]


### Turn 21
def ratio_pair(p, q):
    return sp.Rational(p * q, p * q + p + q)
pairs_info = [(p, q, D, ratio_pair(p, q)) for p, q, D in pairs]
print(pairs_info[:10])

### Turn 22
def minimal_s_for_c(c):
    target = c + 1
    # search s from 3 upwards
    max_s = 1000000  # generous
    best_s = None
    best_ratio = None
    for s in range(3, max_s+1):
        if s % 2 == 1:
            d = 3*s + 2
        else:
            k = s//2
            d = 3*k + 1
        if target % d == 0:
            # found
            best_s = s
            break
    return best_s

c_values = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_values:
    s = minimal_s_for_c(c)
    print(c, s)


### Turn 22
def minimal_s_for_c(c):
    target = c + 1
    max_s = 1000000
    best_s = None
    best_ratio = None
    for s in range(3, max_s + 1):
        if s % 2 == 1:
            d = 3 * s + 2
        else:
            k = s // 2
            d = 3 * k + 1
        if target % d == 0:
            best_s = s
            break
    return best_s
c_values = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_values:
    s = minimal_s_for_c(c)
    print(c, s)

### Turn 23
10162575 // 47


### Turn 23
print(10162575 // 47)

### Turn 24
265710645 // 97


### Turn 24
print(265710645 // 97)

### Turn 25
44636595 // 167


### Turn 25
print(44636595 // 167)

### Turn 26
def gcd_2k_3k1(k):
    return sp.gcd(2*k, 3*k+1)

[gcd_2k_3k1(k) for k in range(1,11)]


### Turn 26
def gcd_2k_3k1(k):
    return sp.gcd(2 * k, 3 * k + 1)
print([gcd_2k_3k1(k) for k in range(1, 11)])

### Turn 27
import math, sys, sympy as sp
A = math.factorial(2025)
A_bitlen = A.bit_length()
A_bitlen


### Turn 27
import math, sys, sympy as sp
A = math.factorial(2025)
A_bitlen = A.bit_length()
print(A_bitlen)

### Turn 28
from fractions import Fraction

def g_from_s(s):
    numer = (A * (2*s)) // (3*s + 2)  # integer floor
    return Fraction(numer, A)  # will be reduced automatically

# compute for each
g0 = Fraction(2,3)
g1 = Fraction(10,3)
g1848374 = g_from_s(16)
g10162574 = g_from_s(15)
g265710644 = g_from_s(64)
g44636594 = g_from_s(55)

g0, g1, g1848374, g10162574, g265710644, g44636594


### Turn 28
from fractions import Fraction

def g_from_s(s):
    numer = A * (2 * s) // (3 * s + 2)
    return Fraction(numer, A)
g0 = Fraction(2, 3)
g1 = Fraction(10, 3)
g1848374 = g_from_s(16)
g10162574 = g_from_s(15)
g265710644 = g_from_s(64)
g44636594 = g_from_s(55)
print((g0, g1, g1848374, g10162574, g265710644, g44636594))

### Turn 29
from fractions import Fraction
total = g0 + g1 + g1848374 + g10162574 + g265710644 + g44636594
total


### Turn 29
from fractions import Fraction
total = g0 + g1 + g1848374 + g10162574 + g265710644 + g44636594
print(total)

### Turn 30
import math
math.gcd(125561848, 19033825)


### Turn 30
import math
print(math.gcd(125561848, 19033825))

### Turn 31
p = 125561848
q = 19033825
mod = 99991
(p+q) % mod


### Turn 31
p = 125561848
q = 19033825
mod = 99991
print((p + q) % mod)

### Turn 32
1848375 % 11


### Turn 32
print(1848375 % 11)

### Turn 33
1848375 % 7


### Turn 33
print(1848375 % 7)

### Turn 34
1848374 % 3


### Turn 34
print(1848374 % 3)

### Turn 35
def D_prime(p,q):
    D = p*q + p + q
    g = sp.gcd(p*q, D)
    return D // g

def possible_pairs(limit):
    res = []
    for p in range(2,limit+1):
        for q in range(p+1,limit+1):
            d = D_prime(p,q)
            if d <= 10:
                res.append((p,q,d, sp.Rational(p*q, D)))
    return res

possible_pairs(20)[:20]


### Turn 35
def D_prime(p, q):
    D = p * q + p + q
    g = sp.gcd(p * q, D)
    return D // g

def possible_pairs(limit):
    res = []
    for p in range(2, limit + 1):
        for q in range(p + 1, limit + 1):
            d = D_prime(p, q)
            if d <= 10:
                res.append((p, q, d, sp.Rational(p * q, D)))
    return res
print(possible_pairs(20)[:20])

### Turn 36
from fractions import Fraction
total = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
total_numer = total.numerator
total_denom = total.denominator
total_numer, total_denom, (total_numer+total_denom)%99991


### Turn 36
from fractions import Fraction
total = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
total_numer = total.numerator
total_denom = total.denominator
print((total_numer, total_denom, (total_numer + total_denom) % 99991))

### Turn 37
def ratio_p2(q):
    return sp.Rational(2*q, 3*q+2)

def ratio_p3(q):
    return sp.Rational(3*q, 4*q+3)

for q in range(3, 30):
    if ratio_p2(q) < ratio_p3(q):
        pass
    else:
        print(q, ratio_p2(q), ratio_p3(q))


### Turn 37
def ratio_p2(q):
    return sp.Rational(2 * q, 3 * q + 2)

def ratio_p3(q):
    return sp.Rational(3 * q, 4 * q + 3)
for q in range(3, 30):
    if ratio_p2(q) < ratio_p3(q):
        pass
    else:
        print(q, ratio_p2(q), ratio_p3(q))

### Turn 38
from fractions import Fraction
total2 = Fraction(4,1) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
total2


### Turn 38
from fractions import Fraction
total2 = Fraction(4, 1) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print(total2)

### Turn 39
(125561848 + 19033825) % 99991


### Turn 39
print((125561848 + 19033825) % 99991)

