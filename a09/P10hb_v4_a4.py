# Auto-extracted from: P10hb_v4.md
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



### Turn 0
import sympy as sp

def f_of_n(n, limit=5000):
    # find smallest m such that there exist three distinct positive divisors of m summing to n
    for m in range(1, limit+1):
        # get divisors of m
        divs = sp.divisors(m)
        # need three distinct
        if len(divs) >= 3:
            # choose any 3 distinct
            # brute
            for i in range(len(divs)):
                for j in range(i+1, len(divs)):
                    for k in range(j+1, len(divs)):
                        if divs[i] + divs[j] + divs[k] == n:
                            return m
    return None

def compute_f(limit_n):
    res = {}
    for n in range(6, limit_n+1):
        res[n] = f_of_n(n, limit=2000) # may need bigger limit
    return res

res=compute_f(30)
res


### Turn 0
import sympy as sp

def f_of_n(n, limit=5000):
    for m in range(1, limit + 1):
        divs = sp.divisors(m)
        if len(divs) >= 3:
            for i in range(len(divs)):
                for j in range(i + 1, len(divs)):
                    for k in range(j + 1, len(divs)):
                        if divs[i] + divs[j] + divs[k] == n:
                            return m
    return None

def compute_f(limit_n):
    res = {}
    for n in range(6, limit_n + 1):
        res[n] = f_of_n(n, limit=2000)
    return res
res = compute_f(30)
print(res)

### Turn 1
def f_bruteforce(n, limit_m=20000):
    # brute compute f(n) up to limit
    for m in range(1, limit_m+1):
        divs = sp.divisors(m)
        if len(divs) >= 3:
            # check any triple sum to n
            for i in range(len(divs)):
                for j in range(i+1, len(divs)):
                    for k in range(j+1, len(divs)):
                        if divs[i] + divs[j] + divs[k] == n:
                            return m
    return None

def f_via_lcm(n, limit=200):
    # search triples a,b,c distinct positive sum n, compute lcm minimal.
    best = None
    besttrip = None
    # generate all triples distinct positive (a<b<c) up to n-3 maybe.
    # Use loops limited due to n.
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            # distinct automatically if a<b<c
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
                besttrip = (a,b,c)
    return best, besttrip

def brute_f(n):
    best,trip = f_via_lcm(n)
    # also compute via brute search of m? but they should match
    return best

for a in range(1,8):
    M=3**a
    bf=brute_f(M)
    print(a,M,bf)


### Turn 1
def f_bruteforce(n, limit_m=20000):
    for m in range(1, limit_m + 1):
        divs = sp.divisors(m)
        if len(divs) >= 3:
            for i in range(len(divs)):
                for j in range(i + 1, len(divs)):
                    for k in range(j + 1, len(divs)):
                        if divs[i] + divs[j] + divs[k] == n:
                            return m
    return None

def f_via_lcm(n, limit=200):
    best = None
    besttrip = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
                besttrip = (a, b, c)
    return (best, besttrip)

def brute_f(n):
    best, trip = f_via_lcm(n)
    return best
for a in range(1, 8):
    M = 3 ** a
    bf = brute_f(M)
    print(a, M, bf)

### Turn 2
def find_triple_for_sum(n):
    best = None
    besttrip = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
                besttrip = (a,b,c)
    return best, besttrip

best,trip = find_triple_for_sum(81)
best,trip


### Turn 2
def find_triple_for_sum(n):
    best = None
    besttrip = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
                besttrip = (a, b, c)
    return (best, besttrip)
best, trip = find_triple_for_sum(81)
print((best, trip))

### Turn 3
def f_min_lcm(n):
    # compute minimal lcm via triple enumeration.
    best = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
    return best

def compute_f_up_to(N):
    arr = {}
    for n in range(6, N+1):
        arr[n] = f_min_lcm(n)
    return arr

arr = compute_f_up_to(200)
ratios = {n: sp.nsimplify(arr[n]/n) for n in arr}
list(ratios.items())[:20]


### Turn 3
def f_min_lcm(n):
    best = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
    return best

def compute_f_up_to(N):
    arr = {}
    for n in range(6, N + 1):
        arr[n] = f_min_lcm(n)
    return arr
arr = compute_f_up_to(200)
ratios = {n: sp.nsimplify(arr[n] / n) for n in arr}
print(list(ratios.items())[:20])

### Turn 4
print("f(45) =", f_min_lcm(45))


### Turn 4
print('f(45) =', f_min_lcm(45))

### Turn 5
print(f_min_lcm(54))


### Turn 5
print(f_min_lcm(54))

### Turn 6
def test_up_to(N):
    for n in range(6, N+1):
        if n%9==0:
            if f_min_lcm(n) != 2*n//3:
                print("counterexample", n, f_min_lcm(n))
                return False
    return True

print(test_up_to(200))


### Turn 6
def test_up_to(N):
    for n in range(6, N + 1):
        if n % 9 == 0:
            if f_min_lcm(n) != 2 * n // 3:
                print('counterexample', n, f_min_lcm(n))
                return False
    return True
print(test_up_to(200))

### Turn 7
def find_best_trip(n):
    best = None
    besttrip = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
                besttrip = (a,b,c)
    return best, besttrip

print(find_best_trip(63))


### Turn 7
def find_best_trip(n):
    best = None
    besttrip = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
                besttrip = (a, b, c)
    return (best, besttrip)
print(find_best_trip(63))

### Turn 8
def compute_L(s):
    # min lcm for sum s
    best = None
    for a in range(1, s):
        for b in range(a+1, s):
            c = s - a - b
            if c <= b:
                continue
            l = sp.ilcm(a,b,c)
            if best is None or l < best:
                best = l
    return best

def compute_ratios(limit=100):
    ratios = {}
    for s in range(6, limit+1):
        L = compute_L(s)
        ratios[s] = sp.Rational(L, s)
    return ratios

ratios = compute_ratios(200)
# find minimal ratio values
min_ratio = min(ratios.values())
min_ratio


### Turn 8
def compute_L(s):
    best = None
    for a in range(1, s):
        for b in range(a + 1, s):
            c = s - a - b
            if c <= b:
                continue
            l = sp.ilcm(a, b, c)
            if best is None or l < best:
                best = l
    return best

def compute_ratios(limit=100):
    ratios = {}
    for s in range(6, limit + 1):
        L = compute_L(s)
        ratios[s] = sp.Rational(L, s)
    return ratios
ratios = compute_ratios(200)
min_ratio = min(ratios.values())
print(min_ratio)

### Turn 9
def compute_min_ratio(limit):
    min_ratio = sp.Rational(1,1)
    min_s = None
    for s in range(6, limit+1):
        L = compute_L(s)
        r = sp.Rational(L, s)
        if r < min_ratio:
            min_ratio = r
            min_s = s
    return min_ratio, min_s

min_ratio, min_s = compute_min_ratio(300)
min_ratio, min_s


### Turn 9
def compute_min_ratio(limit):
    min_ratio = sp.Rational(1, 1)
    min_s = None
    for s in range(6, limit + 1):
        L = compute_L(s)
        r = sp.Rational(L, s)
        if r < min_ratio:
            min_ratio = r
            min_s = s
    return (min_ratio, min_s)
min_ratio, min_s = compute_min_ratio(300)
print((min_ratio, min_s))

### Turn 10
find_best_trip(11)


### Turn 10
print(find_best_trip(11))

### Turn 11
def min_L_and_trip_upto(N):
    best_L = {}
    best_trip = {}
    for s in range(6, N+1):
        best = None
        trip = None
        for a in range(1, s):
            for b in range(a+1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a,b,c)
                if best is None or L < best:
                    best = L
                    trip = (a,b,c)
        best_L[s] = best
        best_trip[s] = trip
    return best_L, best_trip

best_L, best_trip = min_L_and_trip_upto(500)
ratios = {s: sp.Rational(best_L[s], s) for s in best_L}
min_ratio = min(ratios.values())
min_s = min(ratios, key=ratios.get)
min_ratio, min_s, best_trip[min_s]


### Turn 11
def min_L_and_trip_upto(N):
    best_L = {}
    best_trip = {}
    for s in range(6, N + 1):
        best = None
        trip = None
        for a in range(1, s):
            for b in range(a + 1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a, b, c)
                if best is None or L < best:
                    best = L
                    trip = (a, b, c)
        best_L[s] = best
        best_trip[s] = trip
    return (best_L, best_trip)
best_L, best_trip = min_L_and_trip_upto(500)
ratios = {s: sp.Rational(best_L[s], s) for s in best_L}
min_ratio = min(ratios.values())
min_s = min(ratios, key=ratios.get)
print((min_ratio, min_s, best_trip[min_s]))

### Turn 12
def compute_min_ratio_up_to(N):
    min_ratio = sp.Rational(1,1)
    min_s = None
    for s in range(6, N+1):
        best = None
        for a in range(1, s):
            for b in range(a+1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a,b,c)
                if best is None or L < best:
                    best = L
        r = sp.Rational(best, s)
        if r < min_ratio:
            min_ratio = r
            min_s = s
    return min_ratio, min_s

# Let's go up to 200 perhaps; 2000 might be too heavy
print(compute_min_ratio_up_to(200))


### Turn 12
def compute_min_ratio_up_to(N):
    min_ratio = sp.Rational(1, 1)
    min_s = None
    for s in range(6, N + 1):
        best = None
        for a in range(1, s):
            for b in range(a + 1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a, b, c)
                if best is None or L < best:
                    best = L
        r = sp.Rational(best, s)
        if r < min_ratio:
            min_ratio = r
            min_s = s
    return (min_ratio, min_s)
print(compute_min_ratio_up_to(200))

### Turn 13
def compute_R_for_s(limit):
    R = {}
    for s in range(6, limit+1):
        L = compute_L(s)
        R[s] = sp.Rational(L, s)
    return R

R = compute_R_for_s(200)
sorted_R = sorted(R.items(), key=lambda x: x[1])
sorted_R[:10]


### Turn 13
def compute_R_for_s(limit):
    R = {}
    for s in range(6, limit + 1):
        L = compute_L(s)
        R[s] = sp.Rational(L, s)
    return R
R = compute_R_for_s(200)
sorted_R = sorted(R.items(), key=lambda x: x[1])
print(sorted_R[:10])

### Turn 14
def min_R_upto(N):
    best_ratio = sp.Rational(1,1)
    best_s = None
    best_trip = None
    for s in range(6, N+1):
        L, trip = None, None
        best_l = None
        best_t = None
        for a in range(1, s):
            for b in range(a+1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a,b,c)
                if best_l is None or L < best_l:
                    best_l = L
                    best_t = (a,b,c)
        r = sp.Rational(best_l, s)
        if r < best_ratio:
            best_ratio = r
            best_s = s
            best_trip = best_t
    return best_ratio, best_s, best_trip

print(min_R_upto(200))


### Turn 14
def min_R_upto(N):
    best_ratio = sp.Rational(1, 1)
    best_s = None
    best_trip = None
    for s in range(6, N + 1):
        L, trip = (None, None)
        best_l = None
        best_t = None
        for a in range(1, s):
            for b in range(a + 1, s):
                c = s - a - b
                if c <= b:
                    continue
                L = sp.ilcm(a, b, c)
                if best_l is None or L < best_l:
                    best_l = L
                    best_t = (a, b, c)
        r = sp.Rational(best_l, s)
        if r < best_ratio:
            best_ratio = r
            best_s = s
            best_trip = best_t
    return (best_ratio, best_s, best_trip)
print(min_R_upto(200))

### Turn 15
def min_R_upto_fast(N):
    best_ratio = sp.Rational(1,1)
    best_s = None
    best_trip = None
    for a in range(1, N):
        for b in range(a+1, N):
            # compute max c such that a+b+c <= N
            # c will vary to fill sum s = a+b+c, we want s up to N
            # So c can be up to N - a - b
            max_c = N - a - b
            if max_c <= b:
                continue
            # For each possible c > b up to max_c, compute ratio.
            # However we are looking for minimum ratio L/(a+b+c). Since L depends on c.
            # This nested loops could be heavy. Instead we keep earlier method.
    # fallback: original method may be okay for N up to 500 maybe.



### Turn 15
def min_R_upto_fast(N):
    best_ratio = sp.Rational(1, 1)
    best_s = None
    best_trip = None
    for a in range(1, N):
        for b in range(a + 1, N):
            max_c = N - a - b
            if max_c <= b:
                continue

### Turn 16
def min_ratio_trip(limit_a):
    best_ratio = sp.Rational(1,1)
    best_trip = None
    for a in range(1, limit_a+1):
        for b in range(a+1, limit_a+1):
            # bound c up to limit_a as well maybe bigger?
            # We'll restrict c to limit_a as well.
            for c in range(b+1, limit_a+1):
                L = sp.ilcm(a,b,c)
                s = a+b+c
                r = sp.Rational(L, s)
                if r < best_ratio:
                    best_ratio = r
                    best_trip = (a,b,c)
    return best_ratio, best_trip

best_ratio, best_trip = min_ratio_trip(100)
best_ratio, best_trip


### Turn 16
def min_ratio_trip(limit_a):
    best_ratio = sp.Rational(1, 1)
    best_trip = None
    for a in range(1, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            for c in range(b + 1, limit_a + 1):
                L = sp.ilcm(a, b, c)
                s = a + b + c
                r = sp.Rational(L, s)
                if r < best_ratio:
                    best_ratio = r
                    best_trip = (a, b, c)
    return (best_ratio, best_trip)
best_ratio, best_trip = min_ratio_trip(100)
print((best_ratio, best_trip))

### Turn 17
best_ratio, best_trip = min_ratio_trip(200)
best_ratio, best_trip


### Turn 17
best_ratio, best_trip = min_ratio_trip(200)
print((best_ratio, best_trip))

### Turn 18
c_vals = [0, None]  # placeholder
c_vals = [0, 4* (3**2025) ] # this is huge can't compute



### Turn 18
c_vals = [0, None]
c_vals = [0, 4 * 3 ** 2025]
print(c_vals)

### Turn 19
c3 = 1848374
c4 = 10162574
c5 = 265710644
c6 = 44636594
mods = [c % 11 for c in [c3,c4,c5,c6]]
mods


### Turn 19
c3 = 1848374
c4 = 10162574
c5 = 265710644
c6 = 44636594
mods = [c % 11 for c in [c3, c4, c5, c6]]
print(mods)

### Turn 20
def ratio_f_n(n):
    # compute minimal f(n) via search for triple (a,b,c) and divisor s?
    # brute compute f(n) directly using earlier method: minimal lcm among triples with sum n
    # since n not huge.
    return sp.Rational(f_min_lcm(n), n)

ratios_up_to = {}
for n in range(6, 200):
    ratios_up_to[n] = ratio_f_n(n)

# Find minimal ratio among n not divisible by 11
min_ratio_non11 = min([ratios_up_to[n] for n in ratios_up_to if n % 11 != 0])
min_ratio_non11


### Turn 20
def ratio_f_n(n):
    return sp.Rational(f_min_lcm(n), n)
ratios_up_to = {}
for n in range(6, 200):
    ratios_up_to[n] = ratio_f_n(n)
min_ratio_non11 = min([ratios_up_to[n] for n in ratios_up_to if n % 11 != 0])
print(min_ratio_non11)

### Turn 21
def min_ratio_non_multiple(limit):
    best = sp.Rational(1,1)
    best_n = None
    for n in range(6, limit+1):
        if n % 11 == 0:
            continue
        r = sp.Rational(f_min_lcm(n), n)
        if r < best:
            best = r
            best_n = n
    return best, best_n

best_ratio, best_n = min_ratio_non_multiple(500)
best_ratio, best_n


### Turn 21
def min_ratio_non_multiple(limit):
    best = sp.Rational(1, 1)
    best_n = None
    for n in range(6, limit + 1):
        if n % 11 == 0:
            continue
        r = sp.Rational(f_min_lcm(n), n)
        if r < best:
            best = r
            best_n = n
    return (best, best_n)
best_ratio, best_n = min_ratio_non_multiple(500)
print((best_ratio, best_n))

### Turn 22
def min_ratios(limit_a):
    ratios = {}
    for a in range(1, limit_a+1):
        for b in range(a+1, limit_a+1):
            for c in range(b+1, limit_a+1):
                L = sp.ilcm(a,b,c)
                s = a+b+c
                r = sp.Rational(L, s)
                if r not in ratios or L < ratios[r][0]:
                    ratios[r] = (L, (a,b,c))
    # sort ratios
    sorted_ratios = sorted(ratios.items(), key=lambda x: x[0])
    return sorted_ratios

sorted_ratios = min_ratios(50)
[(r, ratios[0]) for r, ratios in sorted_ratios[:10]]


### Turn 22
def min_ratios(limit_a):
    ratios = {}
    for a in range(1, limit_a + 1):
        for b in range(a + 1, limit_a + 1):
            for c in range(b + 1, limit_a + 1):
                L = sp.ilcm(a, b, c)
                s = a + b + c
                r = sp.Rational(L, s)
                if r not in ratios or L < ratios[r][0]:
                    ratios[r] = (L, (a, b, c))
    sorted_ratios = sorted(ratios.items(), key=lambda x: x[0])
    return sorted_ratios
sorted_ratios = min_ratios(50)
print([(r, ratios[0]) for r, ratios in sorted_ratios[:10]])

### Turn 23
c3=1848374
c4=10162574
c5=265710644
c6=44636594
mods7 = [c % 7 for c in [c3,c4,c5,c6]]
mods7


### Turn 23
c3 = 1848374
c4 = 10162574
c5 = 265710644
c6 = 44636594
mods7 = [c % 7 for c in [c3, c4, c5, c6]]
print(mods7)

### Turn 24
ratios_info = sorted_ratios[:10]
ratios_info


### Turn 24
ratios_info = sorted_ratios[:10]
print(ratios_info)

### Turn 25
import sympy as sp

c_vals = [0, 4*sp.Integer(3)**2025, 1848374, 10162574, 265710644, 44636594]  # but can't compute 4M.

# We'll handle c=0 and other small c separately
c_small = [0, 1848374, 10162574, 265710644, 44636594]
def get_small_divisors(n, limit=200):
    # returns set of divisors <= limit
    divs = set()
    for d in sp.divisors(n):
        if d <= limit:
            divs.add(d)
    return divs

for c in c_small:
    divs_c = get_small_divisors(c, limit=200)
    divs_c_plus = get_small_divisors(c+1, limit=200)
    print(c, "divs_c up to 200:", sorted(divs_c))
    print(c, "divs_c_plus up to 200:", sorted(divs_c_plus))


### Turn 25
import sympy as sp
c_vals = [0, 4 * sp.Integer(3) ** 2025, 1848374, 10162574, 265710644, 44636594]
c_small = [0, 1848374, 10162574, 265710644, 44636594]

def get_small_divisors(n, limit=200):
    divs = set()
    for d in sp.divisors(n):
        if d <= limit:
            divs.add(d)
    return divs
for c in c_small:
    divs_c = get_small_divisors(c, limit=200)
    divs_c_plus = get_small_divisors(c + 1, limit=200)
    print(c, 'divs_c up to 200:', sorted(divs_c))
    print(c, 'divs_c_plus up to 200:', sorted(divs_c_plus))

### Turn 26
def max_power_of_3_dividing(n):
    count = 0
    while n % 3 == 0 and n>0:
        n //= 3
        count +=1
    return 3**count

for c in c_small[1:]:
    print(c, max_power_of_3_dividing(c))


### Turn 26
def max_power_of_3_dividing(n):
    count = 0
    while n % 3 == 0 and n > 0:
        n //= 3
        count += 1
    return 3 ** count
for c in c_small[1:]:
    print(c, max_power_of_3_dividing(c))

### Turn 27
def min_L_of_sum(s):
    best = None
    for a in range(1, s):
        for b in range(a+1, s):
            c = s - a - b
            if c <= b:
                continue
            L = sp.ilcm(a,b,c)
            if best is None or L < best:
                best = L
                best_trip = (a,b,c)
    return best, best_trip

def ratio_of_sum(s):
    L, trip = min_L_of_sum(s)
    return sp.Rational(L, s), L, trip

for s in [25,31,53,125,155]:
    r,L,tr = ratio_of_sum(s)
    print(s, r, L, tr)


### Turn 27
def min_L_of_sum(s):
    best = None
    for a in range(1, s):
        for b in range(a + 1, s):
            c = s - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if best is None or L < best:
                best = L
                best_trip = (a, b, c)
    return (best, best_trip)

def ratio_of_sum(s):
    L, trip = min_L_of_sum(s)
    return (sp.Rational(L, s), L, trip)
for s in [25, 31, 53, 125, 155]:
    r, L, tr = ratio_of_sum(s)
    print(s, r, L, tr)

### Turn 28
1848375 % 11


### Turn 28
print(1848375 % 11)

### Turn 29
1848375 % 7


### Turn 29
print(1848375 % 7)

### Turn 30
def f_bruteforce_direct(n, limit=5000):
    best = None
    best_trip = None
    for a in range(1, n):
        for b in range(a+1, n):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a,b,c)
            if best is None or L < best:
                best = L
                best_trip = (a,b,c)
    return best, best_trip

def f_via_divisor_method(n):
    # compute minimal ratio using divisor approach as we propose
    # compute all divisors of n up to some bound maybe
    # We'll search s up to maybe 2000 for simplicity.
    best_ratio = sp.Rational(1,1)
    best_s = None
    best_l = None
    # compute candidate s: divisors of n up to 2000
    for s in sp.divisors(n):
        if s < 6:
            continue
        # compute L(s) minimal via earlier function (limited)
        # For performance we might limit s up to 2000
        if s > 2000:
            continue
        L,_ = min_L_of_sum(s)
        r = sp.Rational(L, s)
        if r < best_ratio:
            best_ratio = r
            best_s = s
            best_l = L
    if best_s is None:
        return None
    return sp.Rational(best_l, s) * n # This returns f estimate but not used

# test small case
k=5
M_small = 3**k
c_test = 10
n = M_small + c_test
f_direct,trip = f_bruteforce_direct(n, limit=2000)
print("n:", n, "direct f:", f_direct, "trip:", trip)


### Turn 30
def f_bruteforce_direct(n, limit=5000):
    best = None
    best_trip = None
    for a in range(1, n):
        for b in range(a + 1, n):
            c = n - a - b
            if c <= b:
                continue
            L = sp.ilcm(a, b, c)
            if best is None or L < best:
                best = L
                best_trip = (a, b, c)
    return (best, best_trip)

def f_via_divisor_method(n):
    best_ratio = sp.Rational(1, 1)
    best_s = None
    best_l = None
    for s in sp.divisors(n):
        if s < 6:
            continue
        if s > 2000:
            continue
        L, _ = min_L_of_sum(s)
        r = sp.Rational(L, s)
        if r < best_ratio:
            best_ratio = r
            best_s = s
            best_l = L
    if best_s is None:
        return None
    return sp.Rational(best_l, s) * n
k = 5
M_small = 3 ** k
c_test = 10
n = M_small + c_test
f_direct, trip = f_bruteforce_direct(n, limit=2000)
print('n:', n, 'direct f:', f_direct, 'trip:', trip)

### Turn 31
n = M_small + 10
divs = sp.divisors(n)
# compute minimal ratio among divisors
best_ratio = sp.Rational(1,1)
best_s = None
for s in divs:
    if s < 6:
        continue
    # compute minimal L for s via brute (if s not too big)
    if s > 200:
        continue
    L,_ = min_L_of_sum(s)
    r = sp.Rational(L, s)
    if r < best_ratio:
        best_ratio = r
        best_s = s
print("best ratio:", best_ratio, "s:", best_s)


### Turn 31
n = M_small + 10
divs = sp.divisors(n)
best_ratio = sp.Rational(1, 1)
best_s = None
for s in divs:
    if s < 6:
        continue
    if s > 200:
        continue
    L, _ = min_L_of_sum(s)
    r = sp.Rational(L, s)
    if r < best_ratio:
        best_ratio = r
        best_s = s
print('best ratio:', best_ratio, 's:', best_s)

### Turn 32
def ratio_of_sum_fast(s):
    # compute ratio for given s via brute enumeration up to s (maybe high). 
    # We'll only compute for s up to maybe 200.
    L,trip = min_L_of_sum(s)
    return sp.Rational(L,s), L, trip

for s in [15,45,135]:
    print(s, ratio_of_sum_fast(s)[0])


### Turn 32
def ratio_of_sum_fast(s):
    L, trip = min_L_of_sum(s)
    return (sp.Rational(L, s), L, trip)
for s in [15, 45, 135]:
    print(s, ratio_of_sum_fast(s)[0])

### Turn 33
def min_ratio_for_c(c):
    A = sp.factorint(c+1)
    # generate all divisors
    divs = sp.divisors(c+1)
    # filter s >=6 and s not divisible by 3
    candidates = [s for s in divs if s >=6 and s % 3 != 0]
    # compute ratio for each s (need minimal L(s)/s)
    # We'll compute L(s) via heuristic: if s divisible by 11, ratio 6/11 (since L(s)=6*(s/11)? Actually scaling base triple 2,3,6 sum 11 gives ratio 6/11, so for any multiple of 11, ratio = 6/11.
    # In general, if s is multiple of base sum, we can scale that base triple.
    # The minimal ratio for s that is multiple of base sum t is the same as ratio for t.
    # So we can precompute base sums and ratios for known optimal triples: 
    # We'll build dict best_ratio_for_sum[t] for t up to maybe 200 (we have). For a multiple m*t, ratio same.
    # However, for s that is not multiple of any base sum with known best ratio, need to compute.
    # But we can approximate by checking known base sums and their multiples.
    # Let's precompute best ratio for sums up to 200 (including scaling). We'll produce a map from sum to ratio.


### Turn 33
def min_ratio_for_c(c):
    A = sp.factorint(c + 1)
    divs = sp.divisors(c + 1)
    candidates = [s for s in divs if s >= 6 and s % 3 != 0]

### Turn 34
def check_divisible(c, s):
    return (c+1) % s == 0

for c in [1848374,10162574,265710644,44636594]:
    print("c:",c)
    for s in [11,7,17,10,13,19,22,23,25,27,31,33,35]:
        if check_divisible(c,s):
            print(" divides",s)


### Turn 34
def check_divisible(c, s):
    return (c + 1) % s == 0
for c in [1848374, 10162574, 265710644, 44636594]:
    print('c:', c)
    for s in [11, 7, 17, 10, 13, 19, 22, 23, 25, 27, 31, 33, 35]:
        if check_divisible(c, s):
            print(' divides', s)

### Turn 35
c = 44636594
divs = sp.divisors(c+1)
sorted_small = [d for d in divs if d <= 200]
sorted_small[:20], len(divs)


### Turn 35
c = 44636594
divs = sp.divisors(c + 1)
sorted_small = [d for d in divs if d <= 200]
print((sorted_small[:20], len(divs)))

### Turn 36
def ratio_of_sum_fast_limit(s, limit=200):
    # compute via brute enumeration up to s (which may be up to 200)
    return ratio_of_sum(s)

for s in [103, 167, 173]:
    print(s, ratio_of_sum_fast_limit(s)[0])


### Turn 36
def ratio_of_sum_fast_limit(s, limit=200):
    return ratio_of_sum(s)
for s in [103, 167, 173]:
    print(s, ratio_of_sum_fast_limit(s)[0])

### Turn 37
float(68/103), float(110/167), float(114/173)


### Turn 37
print((float(68 / 103), float(110 / 167), float(114 / 173)))

### Turn 38
from fractions import Fraction

S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(16,25) + Fraction(2,3) + Fraction(110,167)
S


### Turn 38
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(16, 25) + Fraction(2, 3) + Fraction(110, 167)
print(S)

### Turn 39
p = S.numerator
q = S.denominator
p,q


### Turn 39
p = S.numerator
q = S.denominator
print((p, q))

### Turn 40
(p+q) % 99991


### Turn 40
print((p + q) % 99991)

### Turn 41
c = 265710644
c1 = c+1
sp.factorint(c1)


### Turn 41
c = 265710644
c1 = c + 1
print(sp.factorint(c1))

### Turn 42
c=10162574
sp.factorint(c+1)


### Turn 42
c = 10162574
print(sp.factorint(c + 1))

### Turn 43
sp.factorint(44636595)


### Turn 43
print(sp.factorint(44636595))

### Turn 44
p+q


### Turn 44
print(p + q)

### Turn 45
import math, sys, sympy as sp, time, itertools, functools, math, random, decimal, fractions, collections, itertools, functools, math
A = math.factorial(2025)
len(str(A))


### Turn 45
import math, sys, sympy as sp, time, itertools, functools, math, random, decimal, fractions, collections, itertools, functools, math
A = math.factorial(2025)
print(len(str(A)))

### Turn 46
sp.factorint(265710644)


### Turn 46
print(sp.factorint(265710644))

### Turn 47
def compute_base_ratios(limit=200):
    base = {}
    for s in range(6, limit+1):
        L,_ = min_L_of_sum(s)
        r = sp.Rational(L, s)
        base[s] = r
    # find s where ratio < 2/3
    low = {s:r for s,r in base.items() if r < sp.Rational(2,3)}
    return low

low_s = compute_base_ratios(200)
sorted_low = sorted(low_s.items(), key=lambda x: x[1])
sorted_low[:20]


### Turn 47
def compute_base_ratios(limit=200):
    base = {}
    for s in range(6, limit + 1):
        L, _ = min_L_of_sum(s)
        r = sp.Rational(L, s)
        base[s] = r
    low = {s: r for s, r in base.items() if r < sp.Rational(2, 3)}
    return low
low_s = compute_base_ratios(200)
sorted_low = sorted(low_s.items(), key=lambda x: x[1])
print(sorted_low[:20])

### Turn 48
sorted_low[20:40]


### Turn 48
print(sorted_low[20:40])

### Turn 49
sorted_low[40:80]


### Turn 49
print(sorted_low[40:80])

### Turn 50
sorted_low[80:120]


### Turn 50
print(sorted_low[80:120])

### Turn 51
sorted_low[120:150]


### Turn 51
print(sorted_low[120:150])

### Turn 52
import math, sympy as sp

A = math.factorial(2025)  # Might be huge but pow can handle
# To avoid limit on converting to string, we won't print

def M_mod_s(s):
    # Compute 3^A mod s
    return pow(3, A, s)

def minimal_ratio_for_n(c):
    # compute n = M + c
    # we will consider s up to limit (e.g., 2000)
    limit = 2000
    best_ratio = sp.Rational(1,1)  # start high
    best_s = None
    # Precompute M_mod_s for s up to limit?
    for s in range(6, limit+1):
        # skip if s > 0
        # compute n mod s
        # use M_mod_s (fast)
        mod = (M_mod_s(s) + c % s) % s
        if mod == 0:
            # s divides n
            # compute minimal ratio for sum s, which is minimal L(s)/s (precomputed)
            # We'll compute L(s) directly for s <=2000 via enumeration (slow but okay for limited s)
            L,_ = min_L_of_sum(s)
            r = sp.Rational(L, s)
            if r < best_ratio:
                best_ratio = r
                best_s = s
    return best_ratio, best_s

# Test for c values except c=4M (huge)
c_vals_small = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals_small:
    r,s = minimal_ratio_for_n(c)
    print(c, r, s)


### Turn 52
import math, sympy as sp
A = math.factorial(2025)

def M_mod_s(s):
    return pow(3, A, s)

def minimal_ratio_for_n(c):
    limit = 2000
    best_ratio = sp.Rational(1, 1)
    best_s = None
    for s in range(6, limit + 1):
        mod = (M_mod_s(s) + c % s) % s
        if mod == 0:
            L, _ = min_L_of_sum(s)
            r = sp.Rational(L, s)
            if r < best_ratio:
                best_ratio = r
                best_s = s
    return (best_ratio, best_s)
c_vals_small = [0, 1848374, 10162574, 265710644, 44636594]
for c in c_vals_small:
    r, s = minimal_ratio_for_n(c)
    print(c, r, s)

### Turn 53
c = 10162574
(c+1) % 47


### Turn 53
c = 10162574
print((c + 1) % 47)

### Turn 54
10162575 // 47


### Turn 54
print(10162575 // 47)

### Turn 55
265710645 % 97


### Turn 55
print(265710645 % 97)

### Turn 56
ratio_of_sum(97)


### Turn 56
print(ratio_of_sum(97))

### Turn 57
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S, S.numerator, S.denominator


### Turn 57
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print((S, S.numerator, S.denominator))

### Turn 58
(p+q) % 99991


### Turn 58
print((p + q) % 99991)

### Turn 59
S_old = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(16,25) + Fraction(2,3) + Fraction(110,167)
S_old, S_old == S


### Turn 59
S_old = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(16, 25) + Fraction(2, 3) + Fraction(110, 167)
print((S_old, S_old == S))

### Turn 60
p_old = S_old.numerator
q_old = S_old.denominator
(p_old+q_old) % 99991


### Turn 60
p_old = S_old.numerator
q_old = S_old.denominator
print((p_old + q_old) % 99991)

### Turn 61
10162575 % 11


### Turn 61
print(10162575 % 11)

### Turn 62
10162575 % 7


### Turn 62
print(10162575 % 7)

### Turn 63
10162575 % 13


### Turn 63
print(10162575 % 13)

### Turn 64
10162575 % 17


### Turn 64
print(10162575 % 17)

### Turn 65
10162575 % 25


### Turn 65
print(10162575 % 25)

### Turn 66
265710645 % 11


### Turn 66
print(265710645 % 11)

### Turn 67
265710645 % 7


### Turn 67
print(265710645 % 7)

### Turn 68
265710645 % 13


### Turn 68
print(265710645 % 13)

### Turn 69
265710645 % 17


### Turn 69
print(265710645 % 17)

### Turn 70
sp.factorint(1848375)


### Turn 70
print(sp.factorint(1848375))

### Turn 71
125561848 + 19033825


### Turn 71
print(125561848 + 19033825)

### Turn 72
144595673 % 99991


### Turn 72
print(144595673 % 99991)

### Turn 73
p = 125561848
q = 19033825
(p+q) % 99991


### Turn 73
p = 125561848
q = 19033825
print((p + q) % 99991)

### Turn 74
import fractions, math
from fractions import Fraction
S = Fraction(2,3) + Fraction(10,3) + Fraction(16,25) + Fraction(30,47) + Fraction(64,97) + Fraction(110,167)
S


### Turn 74
import fractions, math
from fractions import Fraction
S = Fraction(2, 3) + Fraction(10, 3) + Fraction(16, 25) + Fraction(30, 47) + Fraction(64, 97) + Fraction(110, 167)
print(S)

### Turn 75
from sympy import Rational
S_sym = Rational(2,3) + Rational(10,3) + Rational(16,25) + Rational(30,47) + Rational(64,97) + Rational(110,167)
S_sym


### Turn 75
from sympy import Rational
S_sym = Rational(2, 3) + Rational(10, 3) + Rational(16, 25) + Rational(30, 47) + Rational(64, 97) + Rational(110, 167)
print(S_sym)

### Turn 76
int(S_sym.p + S_sym.q) % 99991


### Turn 76
print(int(S_sym.p + S_sym.q) % 99991)

### Turn 77
c_vals = [1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    fac = sp.factorint(c+1)
    print(c, fac)


### Turn 77
c_vals = [1848374, 10162574, 265710644, 44636594]
for c in c_vals:
    fac = sp.factorint(c + 1)
    print(c, fac)

### Turn 78
def minimal_ratio_n(c):
    # compute minimal ratio r = f(n)/n for n = M + c, using small s up to 2000
    # We'll use condition: s|n iff (M_mod_s + c% s) % s == 0
    limit = 2000
    best = sp.Rational(1,1)
    best_s = None
    for s in range(6, limit+1):
        # compute M mod s
        if math.gcd(s,3) == 1:
            M_mod = 1  # since A multiple of phi(s)
        else:
            M_mod = pow(3, A, s)
        if (M_mod + c % s) % s == 0:
            # s divides n
            # compute minimal ratio for sum s: we can compute using known base: r_min_s = min_{t|s} r(t)
            # We'll compute r for s directly using previously computed min_L_of_sum(s)
            L,_ = min_L_of_sum(s)
            r = sp.Rational(L, s)
            # But there may exist divisor t of s with smaller ratio, but that t would also divide n,
            # and we will consider it in loop when s=t. So we can just use r.
            if r < best:
                best = r
                best_s = s
    return best, best_s

for c in [0, 1848374, 10162574, 265710644, 44636594]:
    r,s = minimal_ratio_n(c)
    print(c, r, s)


### Turn 78
def minimal_ratio_n(c):
    limit = 2000
    best = sp.Rational(1, 1)
    best_s = None
    for s in range(6, limit + 1):
        if math.gcd(s, 3) == 1:
            M_mod = 1
        else:
            M_mod = pow(3, A, s)
        if (M_mod + c % s) % s == 0:
            L, _ = min_L_of_sum(s)
            r = sp.Rational(L, s)
            if r < best:
                best = r
                best_s = s
    return (best, best_s)
for c in [0, 1848374, 10162574, 265710644, 44636594]:
    r, s = minimal_ratio_n(c)
    print(c, r, s)

