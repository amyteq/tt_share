%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

import os
import sys
import subprocess
import warnings
warnings.simplefilter('ignore')

def set_env(input_archive, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    subprocess.run([
        sys.executable, 
        '-m', 
        'pip', 
        'install', 
        '--no-index', 
        '--find-links', 
        f'{temp_dir}/wheels', 
        'unsloth', 
        'trl', 
        'vllm', 
        'openai_harmony',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz', 
    temp_dir='/kaggle/tmp/setup'
)

subprocess.run(['ls', '/kaggle/tmp/setup/tiktoken_encodings'])

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'
os.environ['VLLM_BATCH_INVARIANT'] = '1'

import gc
import re
import json
import math
import time
import queue
import threading
import contextlib

from __future__ import annotations
from collections import deque, Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

# global constants
ERROR_TIMEOUT = '[ERROR] TIMEOUT:'
SANDBOX_TESTED = False
SANDBOX_PROMPT = 'SANDBOX TEST>'

class CFG:

    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        
        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'
        
        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'
        
        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'
        
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        
        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        
        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )

    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Symbolic differentiation and integration\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n'
        '- Working with mathematical expressions symbolically\n\n'
        
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations for large datasets\n'
        '- Matrix operations and eigenvalue problems\n'
        '- Statistical computations\n\n'
        
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n'
        '- Basic operations for single values\n\n'
        
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification and large-scale computation\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Document your computational strategy clearly\n'
        '- Validate computational results against known cases or theoretical bounds'
    )

    # --- Unified boosts by tag (integrate old CHEAT_SHEET_BOOSTS + preference_prompt sections) ---
    boost_by_tag = {
        "NT": (
            "NT BOOST (pick only what applies; keep it concise):\n"
            "- Modular arithmetic / CRT; split prime powers.\n"
            "- Huge exponent B^(k!): for many small d with gcd(d,B)=1, often B^(k!) ≡ 1 (mod d).\n"
            "- Valuations/LTE/Legendre when factorials appear.\n"
            "\n"
            "NT TEMPLATE (LCM-under-sum problems; concrete recipe):\n"
            "- Sanity lower bound for 'divisors sum = n': if a,b,c are divisors of X, then a,b,c <= X so n=a+b+c <= 3X.\n"
            "  Hence f(n) >= n/3 and f(n)/n >= 1/3. If your computed ratio < 1/3 or implies g(c)=0, something is wrong.\n"
            "- Re-model: for a target sum n, treat candidates as choosing 3 divisors a<b<c with a+b+c=n;\n"
            "  a,b,c | X ⇒ minimal feasible X is lcm(a,b,c). So search for small-sum patterns minimizing lcm/sum.\n"
            "- Do a SMALL Python discovery first (n<=200): compute best ratios r(S)=f(S)/S and keep top candidates (S, r(S)).\n"
            "- Use scaling: if S | N, then f(N) <= (N/S)*f(S), so f(N)/N <= r(S). Thus per-instance, scan only S|N from the shortlist.\n"
            "- If N = B^K + c with factorial-like K: for gcd(S,B)=1 and K multiple of ord/phi,\n"
            "  B^K ≡ 1 (mod S) ⇒ N ≡ 1+c (mod S) ⇒ S|N ⇔ S|(c+1). This turns feasibility into small-divisor checks on (c+1).\n"
            "- g(c) is normalized by M (not by N=M+c). If c = t*M, then N/M = (t+1) multiplies the value.\n"
            "  Sanity check: if you found f(N)≈r*N for both N=M and N=5M, then g(4M) must be ≈5*g(0).\n"
            "- Once each g(c) is locked, aggregate with Fraction/Rational and take mod.\n\n"

            "NORMALIZATION RULE (MANDATORY):\n"
            "- If you normalize by a fixed base M but evaluate at N=M+c, do NOT confuse f(N)/N with f(N)/M.\n"
            "- If you find a ratio r = f(N)/N, then g(c)=f(N)/M = (N/M)*r. If c=t*M, multiplier is (t+1).\n\n"
            "NT RED LINES (avoid common false shortcuts):\n"
            "- Do NOT reduce to 'min prime factor of (n-1)' or similar heuristics unless brute-verified for n<=200.\n"
            "- Always distinguish f(N)/N vs f(N)/M when normalization base M is fixed; if c=t*M then scale by (t+1).\n"
        ),
        "ALG": (
            "ALG BOOST (pick only what applies; keep concise):\n\n"

            "General algebra discipline:\n"
            "- Before heavy manipulation: test small cases / sanity constraints; look for invariants and equality/degenerate cases.\n"
            "- Keep expressions exact when possible (Fraction/Rational), avoid floating drift.\n\n"

            "Inequalities:\n"
            "- Check homogeneity/symmetry; normalize (e.g., sum=1) only if valid; identify equality cases early.\n"
            "- Common tools: AM-GM, C-S (Titu), rearrangement, Chebyshev; use Jensen/Karamata only after verifying convexity and domain.\n"
            "- For cyclic sums: try substitution, smoothing, uvw (symmetric polynomials) when appropriate.\n\n"

            "Functional equations:\n"
            "- Try special inputs: 0, 1, -1; symmetry pairs (x,1-x), (x,1/x), (x,-x), etc.\n"
            "- Prove injective/surjective/boundedness; derive recursion/iteration; watch for hidden domain restrictions.\n"
            "- If polynomial-like: compare two expressions to force linearity/additivity; use Cauchy-style constraints with regularity if given.\n\n"

            "Polynomials:\n"
            "- Vieta, factor by roots, remainder theorem; compare coefficients; consider resultants/derivatives for repeated roots.\n"
            "- For integer/rational roots: Rational Root Theorem; mod arguments; use gcd with x^n±1 when cyclotomic structure appears.\n\n"

            "Sequences / transforms (algebraic viewpoint):\n"
            "- Convolution/correlation/shift operators often correspond to generating functions.\n"
            "  Use A(x)=Σ a_i x^i and B(x)=Σ b_i x^i (Laurent if negative indices appear).\n"
            "- Distinguish convolution A(x)B(x) vs correlation A(x)B(x^{-1}); verify which matches the definition.\n"
            "- Use roots of unity / factorization / linear algebra constraints; then brute-check candidate constructions in Python.\n"
        ),
        "COMB": (
            "COMB BOOST (pick only what applies; keep concise):\n"
            "- If the statement asks “how many / count / number of”, treat it as a counting + structure problem: derive constraints, then enumerate/verify.\n"
            "\n"
            "Shift / correlation pattern (very common):\n"
            "- When you see shifts S_n and a sum like Σ_t α(t+n)β(t), interpret it as *discrete correlation* (not convolution).\n"
            "- Represent finitely-supported sequences as polynomials / Laurent polynomials:\n"
            "  A(x)=Σ a_i x^i,  B(x)=Σ b_i x^i  (if β has negative indices, allow Laurent powers).\n"
            "- Correlation identity corresponds to coefficients of A(x)·B(x^{-1}). If it has exactly two nonzero shifts,\n"
            "  then A(x)·B(x^{-1}) has exactly two nonzero monomials.\n"
            "- Units/normalization: because Laurent units are ±x^k, you can shift/scale so the target becomes 1+x^d.\n"
            "  (Do NOT confuse this with A(x)·B(x).)\n"
            "- Key pitfall: you cannot choose irreducible/cyclotomic factors arbitrarily. A(x) must be a divisor of a *single* 1+x^d (up to ±x^k).\n"
            "- Practical method: enumerate feasible d (bounded), factor 1+x^d, generate candidate divisors A, shift to the allowed support window,\n"
            "  deduplicate, and verify by constructing β. Increase the d-bound until the count stabilizes twice.\n"
            "- Do not count candidates only by factor/subset logic. You must construct β and verify the correlation sequence has exactly two 1’s and 0 elsewhere (scan n in a safe finite range covering supports).\n"
            "\n"
            "General combinatorics toolkit:\n"
            "- Invariants/monovariants: parity/mod invariants, potential functions, energy arguments.\n"
            "- Pigeonhole: map objects→boxes; optimize collisions; extremal choice.\n"
            "- Counting: double count; bijection; inclusion–exclusion; recursion/generating functions if appropriate.\n"
            "- Graph/extremal: handshake lemma, average degree, extremal element arguments.\n"
            "\n"
            "Verification discipline:\n"
            "- Before finalizing, brute-check small cases / bounds in Python, and run a stability check if you used a truncation/enumeration bound.\n"
        ),
        "GEO": (
            "GEO BOOST (pick only what applies; keep concise):\n\n"

            "Angle / cyclic structure:\n"
            "- Start with angle chase; look for cyclic quadrilaterals; use directed angles mod 180° for robustness.\n"
            "- If many equal angles: suspect spiral similarity / Miquel point / cyclicity.\n\n"

            "Similarity / ratios:\n"
            "- Build similarity chains; use homothety for circle tangency / parallel chords.\n"
            "- Convert length ratios to power-of-a-point when circles are present; use Menelaus/Ceva for collinearity/concurrency.\n\n"

            "Circles:\n"
            "- Power of a point, radical axis/center: products of segments, tangency, coaxality.\n"
            "- If two circles appear: try radical axis; if three circles: radical center.\n\n"

            "Inversion / transformations:\n"
            "- Use inversion when tangency/intersection patterns repeat; choose center at a key point (tangency or intersection).\n"
            "- Map circles↔lines; preserve angles; simplify to collinearity or power-of-a-point.\n"
            "- Also consider reflection, rotation, or spiral similarity as lighter alternatives.\n\n"

            "Coordinates (use only if chase stalls):\n"
            "- Complex plane for cyclic configurations; barycentric for triangle-centric problems; vectors for midpoints/parallels.\n"
            "- After coordinates, reduce to a small algebraic identity and verify with a random numeric check (then prove).\n"
        ),
        # UNK intentionally has no boost (let the model proceed normally)
    }

    # discovery_prompt
    discovery_by_tag = {
        "NT0": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the two blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n\n"
            "BLOCK 1: RATIO_TABLE\n"
            "- Using Python, compute f(n) for all 6<=n<=200 by enumerating triples a<b<c with a+b+c=n and minimizing lcm(a,b,c).\n"
            "- Store best ratio r(n)=f(n)/n as Fraction, keep TOP 20 with a witness triple.\n"
            "- Sanity: enforce r(n) >= 1/3 (since n=a+b+c<=3*f(n)). If violated, fix your routine.\n"
            "- Print exactly 20 lines:  n | r(n) | (a,b,c)\n\n"
            "BLOCK 2: PER_INSTANCE\n"
            "For each queried c in the problem:\n"
            "- Let D=c+1.\n"
            "- Enumerate divisors d of D with d<=200 and gcd(d,3)=1.\n"
            "- Pick chosen_d with minimal r(chosen_d) from RATIO_TABLE.\n"
            "- Print ONE line per c exactly:\n"
            "  c | D | chosen_d | r(chosen_d) | witness | assert(D%chosen_d==0)\n"
            "- Scaling sanity: if c=t*M (exact multiple), compute s=N/M=t+1 explicitly and use g(c)=s*r.\n\n"
            "Finally (still in this turn): aggregate all g(c) using Fraction/Rational, reduce to p/q, compute (p+q)%99991.\n"
            "Print: p q mod\n\n"
            "Write exactly this token on a new line and STOP: END_OF_DISCOVERY\n"
        ),
        "NT": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n"
            "Use exact arithmetic (Fraction/Rational). Keep prints compact.\n\n"

            "BLOCK 0: PARSE (1 line)\n"
            "- Identify the instance list/parameters to evaluate (e.g., values of c, or cases like c=t*M). "
            "State them in one line: INSTANCES: ...\n\n"

            "BLOCK 1: RATIO_TABLE (local optimal patterns)\n"
            "- Goal: discover a small shortlist of best ratios r(n)=f(n)/n via bounded brute force (n<=Nmax).\n"
            "- Using Python, compute f(n) for all 6<=n<=200 by enumerating triples a<b<c with a+b+c=n and minimizing lcm(a,b,c).\n"
            "- Store best r(n)=f(n)/n as Fraction and a witness triple (a,b,c). Keep TOP 20 by r(n).\n"
            "- Sanity checks (MANDATORY):\n"
            "  * Lower bound: r(n) >= 1/3 since n=a+b+c <= 3*f(n).\n"
            "  * Spot-check: randomly verify a few n by re-running enumeration to ensure stability.\n"
            "- Print exactly 20 lines:  n | r(n) | (a,b,c)\n\n"

            "BLOCK 2: FEASIBILITY REDUCTION (generic)\n"
            "- Derive the modular/divisibility criterion that links your instances to candidate n from RATIO_TABLE.\n"
            "- If the problem has a huge term like B^K + c: for any d with gcd(d,B)=1 and K multiple of ord/phi, "
            "often B^K ≡ 1 (mod d), so feasibility reduces to d | (c+1). Use this ONLY when justified.\n"
            "- If no such reduction is valid, state: REDUCTION: NONE and proceed by direct bounded checks.\n"
            "Print one line: REDUCTION: ...\n\n"

            "BLOCK 3: PER_INSTANCE (evidence required)\n"
            "For each instance parameter c in the problem:\n"
            "- Define the reduced target D (e.g., D=c+1 if REDUCTION says so; otherwise state D=N/A).\n"
            "- Enumerate candidate divisors d of D with d<=200 and gcd(d,3)=1 (or the appropriate gcd filter for base B).\n"
            "- Among candidate d that appear in RATIO_TABLE, pick chosen_d with minimal r(chosen_d).\n"
            "- Print ONE evidence line per instance exactly:\n"
            "  c | D | chosen_d | r(chosen_d) | witness | assert(D%chosen_d==0)\n"
            "- If c is an exact multiple of the base scale M (c=t*M), compute the scale factor s=N/M=t+1 explicitly.\n"
            "  NEVER confuse f(N)/N with f(N)/M: g(c)=f(N)/M = s * (f(N)/N) ≈ s*r(chosen_d).\n\n"

            "BLOCK 4: AGGREGATE (exact)\n"
            "- Compute each required quantity g(·) exactly as a Fraction/Rational and aggregate exactly.\n"
            "- Reduce the final result to a canonical form (e.g., integer; or p/q in lowest terms) as required by the problem.\n"
            "- If the problem asks for a remainder/modulus, apply it at the very end exactly as specified.\n"
            "- Print the minimal final artifacts needed for downstream: (i) the exact reduced value (integer or p/q), and (ii) the final answer.\n\n"
            "- Do NOT guess the final step; explicitly compute it with Python using exact arithmetic.\n"
        ),
        "ALG": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n"
            "Use exact arithmetic where possible (Fraction/Rational). Keep prints compact.\n\n"

            "BLOCK 0: PARSE (1-3 lines)\n"
            "- Restate variables/domains/constraints. List what must be proven/computed.\n\n"

            "BLOCK 1: SMALL-CASE / SANITY (MANDATORY)\n"
            "- Evaluate/check small cases or random numeric instances to detect structure, constraints, or counterexamples.\n"
            "- Print: 3-10 tested cases + observed invariants/patterns.\n\n"

            "BLOCK 2: CANDIDATE FORM (if applicable)\n"
            "- If it looks like FE: hypothesize form (linear/affine/multiplicative), based on tests.\n"
            "- If it looks like polynomial: guess degree/roots/factor shape from values/constraints.\n"
            "- If it looks like inequality: identify likely equality case(s) from tests and whether it is homogeneous.\n"
            "- Print one line: HYPOTHESIS: ... (or NONE)\n\n"

            "BLOCK 3: TOOL VERIFICATION PLAN\n"
            "- Provide a minimal Python check you will use later (symbolic simplify / solve / brute for bounded range).\n"
            "- Print one line: VERIFY: ...\n\n"

            "Write exactly this token on a new line and STOP: END_OF_DISCOVERY\n"
        ),

        "COMB": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n"
            "Prefer generating functions / counting with verification by brute for small sizes.\n\n"

            "BLOCK 0: PARSE (1-3 lines)\n"
            "- Restate objects counted and constraints. Identify parameters and feasible ranges.\n\n"

            "BLOCK 1: SMALL ENUMERATION (MANDATORY)\n"
            "- Use Python to brute small parameter sizes (as small as possible but informative).\n"
            "- Print a table of counts for 5-15 small instances.\n\n"

            "BLOCK 2: PATTERN / RECURSION CANDIDATES\n"
            "- From the table, propose 1-3 candidate patterns: recurrence, closed form, generating function, bijection idea.\n"
            "- Print: CANDIDATES: ...\n\n"

            "BLOCK 3: DISCRIMINATION TEST\n"
            "- Implement checks that distinguish candidates (extra brute sizes; invariant checks; constructive existence).\n"
            "- Print: which candidate(s) survive.\n\n"
            
            "Write exactly this token on a new line and STOP: END_OF_DISCOVERY\n"
        ),

        "GEO": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n"
            "Goal: extract configuration + likely lemmas; use numeric sanity via coordinates if stuck.\n\n"

            "BLOCK 0: PARSE CONFIG\n"
            "- List given objects (points/lines/circles), key relations (parallel/perpendicular/tangent/cyclic), and target.\n\n"
            
            "BLOCK 1: KEY STRUCTURE HYPOTHESES\n"
            "- Suggest 2-4 plausible tools: cyclic/angles, similarity/homothety, power/radical axis, inversion, coordinates.\n"
            "- For each, state WHAT you would try to prove/construct (e.g., show cyclic, find spiral center, find radical axis).\n\n"
            
            "BLOCK 2: QUICK NUMERIC CHECK (optional but recommended if claim-based)\n"
            "- If appropriate, set up a coordinate model (vectors/complex) for one random non-degenerate instance and sanity-check the claim.\n"
            "- Print: whether the claim holds numerically in that instance (YES/NO/INCONCLUSIVE).\n\n"

            "Write exactly this token on a new line and STOP: END_OF_DISCOVERY\n"
        ),

        "UNK": (
            "STRICT DISCOVERY (TURN 1 ONLY). Output ONLY the blocks below, no narrative.\n"
            "Do NOT write \\boxed{} or 'Answer:' in this turn.\n"
            "Goal: identify the right lens (NT/ALG/COMB/GEO) via minimal tests.\n\n"

            "BLOCK 0: QUICK TRIAGE\n"
            "- List top 2 suspected categories with 1 reason each.\n\n"
            
            "BLOCK 1: 2 MINIMAL TESTS (MANDATORY)\n"
            "- Run two cheap checks that reveal structure:\n"
            "  (i) small-case brute / numeric sampling; (ii) algebraic simplification / modular check / geometric numeric check.\n"
            "- Print results of both tests.\n\n"
            
            "BLOCK 2: CHOOSE TAG\n"
            "- Decide a working tag among NT/ALG/COMB/GEO (or keep UNK if truly unclear) and state next-step plan in 1-3 bullets.\n\n"
            
            "Write exactly this token on a new line and STOP: END_OF_DISCOVERY\n"
        ),
    }
    
    # sanity_check_prompt
    sanity_by_tag = {
        "NT": (
            "SCALE SANITY CHECK (MANDATORY):\n"
            "Before finalizing, explicitly verify normalization by base M:\n"
            "1) Identify any c that is an exact multiple of M (c=t*M).\n"
            "2) Compute s=N/M=t+1 and show how g(c)=s*r (or compute g(c) directly as f(N)/M).\n"
            "3) Recompute the final sum using exact Fractions.\n"
            "Then output the final integer answer."
        )
    }

    # count the final answer as python output in turns
    py_verify_enabled = False
    py_verify_weight = 0.35

    sanity_gate_enabled = False
    sanity_gate_max_tries = 1

    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    vllm_port = 8000
    vllm_base_url = 'http://localhost:8000/v1'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 4
    batches = 1 # 2 batches: 4 + 4 or 3 + 5, explorers + conquerors
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02

    # AST validate & fix
    ast_fix_slice = True
    ast_fix_print = True

    sandbox_helper_level = 2
    sandbox_helper_codes = ''
    entropy_computer_ver = 'v1d5'

    # monkey-patches
    mp_recursion_limit = 2000
    mp_print_max_digits = 100
    mp_print_max_strsize = 1000
    mp_print_max_seqsize = 100
    
    # tool output truncation
    tool_output_max_chars = 8000
    tool_output_head = 2000
    tool_output_tail = 2000
    tool_output_max_lines = 200

    # debug
    debug = False
    debug_req = True
    debug_req_full = 20
    debug_resp = True
    debug_limit = 3000

set_seed(CFG.seed)

import ast

def _extract_names(t: ast.AST):
    """Return a list of ast.Name (Load ctx) that are safe to print."""
    if isinstance(t, ast.Name):
        return [ast.Name(id=t.id, ctx=ast.Load())]
    if isinstance(t, (ast.Tuple, ast.List)):
        out = []
        for e in t.elts:
            out.extend(_extract_names(e))
        return out
    return []

def _is_print_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "print"
    )

def _fix_print(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not CFG.ast_fix_print:
        return code

    last = tree.body[-1]

    # 1) Last is expression: replace with print(expr) (avoid double execution)
    if isinstance(last, ast.Expr) and not _is_print_stmt(last):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last.value],
                keywords=[],
            )
        )

    # 2) Last is assignment: keep it, then print safe targets (Names only)
    elif isinstance(last, ast.Assign):
        args = []
        for t in last.targets:
            args.extend(_extract_names(t))
        if args:  # only add if we have something safe to print
            tree.body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=args,
                        keywords=[],
                    )
                )
            )

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

class _Rewriter(ast.NodeTransformer):
    # rewrite: x[:n] -> safe_head(x, n)
    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.lower is None and sl.step is None and sl.upper is not None:
            return ast.Call(
                func=ast.Name(id="safe_head", ctx=ast.Load()),
                args=[node.value, sl.upper],
                keywords=[],
            )
        return node

def _fix_slice(code: str) -> str:
    if not code or not code.strip() or not CFG.ast_fix_slice:
        return code
    try:
        tree = ast.parse(code)
        tree = _Rewriter().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
    except Exception:
        pass
    return code

def _rewrite_code(code: str) -> str:
    code = _fix_slice(code)
    return _fix_print(code)

# for object, use hasattr/getattr; for dict, use in/d[k]!
def _attr(obj: Any, *keys, default=None):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                v = obj[k]
                if v is None: continue
                return v
    else:
        for k in keys:
            if hasattr(obj, k):
                v = getattr(obj, k)
                if v is None: continue
                return v
    return default

def _hasattr(obj: Any, name: str):
    return hasattr(obj, name) or (isinstance(obj, dict) and name in obj)

def _to_dict(obj) -> dict:
    d = vars(obj)
    return {k: v for k, v in d.items() if not k.startswith("_") and not callable(v)}

### other utils
def _fmt_time(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def _format_markdown(text: str, mode: str = "quote") -> str:
    if not text:
        return ""
    text = text.strip('\n')
    lines = text.split('\n')
    escaped_lines = [f"\\{line}" if line.startswith('#') and mode == 'quote' else line for line in lines]
    processed_text = '\n'.join(escaped_lines)
    if mode in ["markdown", "text", "python", "json"]:
        return f"```{mode}\n{processed_text}\n```\n"
    if mode == "quote":
        return '\n'.join([f"> {line}" for line in escaped_lines]) + "\n"
    if mode == "":
        return processed_text + "\n"
    return f"```\n{processed_text}\n```\n"

def _delete(name: str):
    if name is not None and name != "" and name in globals(): 
        del globals()[name]

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def stat_summary(detailed_results: list[dict]):
    if not detailed_results:
        return []

    metrics_to_process = [
        'Response Length', 
        'Python Calls', 
        'Python Errors', 
        'Timeout Errors', 
        '_Elapsed'
    ]
    
    num_attempts = len(detailed_results)
    stats_list = []

    raw_totals = {}
    for metric in metrics_to_process:
        values = [res.get(metric, 0) for res in detailed_results]
        total_val = sum(values)
        raw_totals[metric] = total_val
        avg_val = total_val / num_attempts if num_attempts > 0 else 0

        if metric == '_Elapsed':
            total_disp = _fmt_time(total_val)
            avg_disp = _fmt_time(avg_val)
        else:
            # avg_val = round(avg_val, 2)
            total_disp = str(total_val)
            avg_disp = str(int(avg_val))

        stats_list.append({
            'metric': metric,
            'total': total_disp,
            'avg': avg_disp
        })

    # TPS: Tokens per Second
    total_tokens = raw_totals.get('Response Length', 0)
    total_elapsed = raw_totals.get('_Elapsed', 0)
    avg_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0
    stats_list.append({
        'metric': 'Tokens/Sec (TPS)',
        'total': '-', 
        'avg': f"{avg_tps:.2f}"
    })

    return stats_list

class AIMO3Logger:
    def __init__(self, cfg, rag):
        self.cfg = cfg
        self.rag = rag

    def get_debug_snippet(self, text: str) -> str:
        limit = self.cfg.debug_limit
        if not text or len(text) <= limit:
            return text or ""
        head = text[:100]
        tail_len = limit - 100
        tail = text[-tail_len:]
        return f"{head}\n ... \n{tail}"

    def log_planner_block(self, plan_raw: str, plan_sanitized: str, plan_digest: str) -> str:
        raw_snip = self.get_debug_snippet(plan_raw)
        san_snip = self.get_debug_snippet(plan_sanitized)
        digest = plan_digest.strip()

        out = []
        out.append("### Planner Output (raw)\n")
        out.append(_format_markdown(raw_snip, mode='text'))
        out.append("### Planner Output (sanitized)\n")
        out.append(_format_markdown(san_snip, mode='text'))
        out.append("### Plan Digest\n")
        out.append(_format_markdown(digest, mode='text'))
        return "".join(out)

    def write_debug_logs(self, detailed_results, df_votes: pd.DataFrame, problem, problem_id="UNK", problem_time=""):
        if not self.cfg.debug:
            return
        try:
            summary_lines = ["\n## Summary Stats\n"]
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                cols = [c for c in df.columns if not c.startswith('_')]
                summary_lines.append(df[cols].to_markdown(index=False))
                summary_lines.append("\n\n")
                
                stat_df = pd.DataFrame(stat_summary(detailed_results))
                summary_lines.append(stat_df.to_markdown(index=False))
                summary_lines.append("\n\n")

            if not df_votes.empty:
                summary_lines.append("## Vote Counts\n")
                summary_lines.append(df_votes.to_markdown(index=False))
                summary_lines.append("\n")

            final_log_content = [f"# Problem ID: {problem_id}\n"]
            final_log_content.append(f"Problem spent time: **{problem_time}**\n\n")
            final_log_content.append(f"**Problem:**\n{_format_markdown(problem)}\n")
            cfg_json = _format_markdown(json.dumps(_to_dict(self.cfg), cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2), 'json')
            final_log_content.append(f"**CFG:**\n{cfg_json}\n\n")
            rag_json = _format_markdown(json.dumps(_to_dict(self.rag), cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2), 'json')
            final_log_content.append(f"**RAG:**\n{rag_json}\n\n")
            final_log_content.extend(summary_lines)
            final_log_content.append("\n===\n")

            sorted_results = sorted(detailed_results, key=lambda x: x['Attempt'])
            for res in sorted_results:
                log_content = res.get('_Log', '')
                if log_content:
                    final_log_content.append(log_content)
                    final_log_content.append("\n===\n")

            output_path = f"{problem_id}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(final_log_content))
            print(f"Debug log written to {output_path}")
        except Exception as e:
            print(f"Failed to write debug log: {e}")

class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self._helper_codes = [
# Level 0: minimum imports
"""
import builtins, math, sys, re, itertools, collections
import fractions
import numpy
import networkx
import scipy
import sympy
import sympy as sp
import mpmath

from fractions import Fraction
from functools import lru_cache
from collections import Counter, deque, defaultdict
from itertools import combinations, permutations, product, count
from sympy import cyclotomic_poly, isprime, primerange, factorint
from sympy.ntheory import multiplicity, n_order
from sympy.ntheory.modular import crt
from sympy.ntheory.residue_ntheory import primitive_root, nthroot_mod

mpmath.mp.dps = 64
""",
# Level 1: safe preambles, monkey-patches
# sympy.n_order == sympy.ntheory.n_order == sympy.ntheory.residue_ntheory.n_order already
# math.gcd/lcm is faster (C, support int only) than sympy.gcd/lcm (python, support int & symbols)
# networkx, scipy won't be used by vllm model anyway?!
# NOTE: we shall use {{...}} and \\n inside the sandbox code itself
f"""
RECURSION_LIMIT = {CFG.mp_recursion_limit}
MAX_DIGITS = {CFG.mp_print_max_digits}
MAX_STRSIZE = {CFG.mp_print_max_strsize}
MAX_SEQSIZE = {CFG.mp_print_max_seqsize}
BIT_THRESHOLD = int(MAX_DIGITS * 3.3219)

sys.setrecursionlimit(RECURSION_LIMIT)
sys.set_int_max_str_digits(0)

sympy.crt = sp.crt = crt
sympy.mod = sp.mod = sympy.Mod
sympy.norder = sp.norder = n_order
sympy.multiplicative_order = sp.multiplicative_order = n_order
sympy.npolycyclotomic = cyclotomic_poly
math.isprime = sympy.isprime

_orig_sympy_gcd = sympy.gcd
_orig_sympy_lcm = sympy.lcm

def optimized_gcd(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return math.gcd(a, b)
    return sympy._orig_gcd(a, b)

def optimized_lcm(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return math.lcm(a, b)
    return sympy._orig_lcm(a, b)

if not hasattr(sympy, "_orig_gcd"):
    sympy._orig_gcd = sympy.gcd
    sympy.gcd = optimized_gcd

if not hasattr(sympy, "_orig_lcm"):
    sympy._orig_lcm = sympy.lcm
    sympy.lcm = optimized_lcm

def _bit_length(self): return int(self).bit_length()
sympy.Integer.bit_length = _bit_length
sympy.core.numbers.Integer.bit_length = _bit_length
sympy.Integer.__and__ = lambda self, other: int(self) & int(other)
sympy.Integer.__or__ = lambda self, other: int(self) | int(other)

def patched_primerange(a, b=None, step=None):
    if b is None:
        return sympy._orig_primerange(a)
    return sympy._orig_primerange(a, b)
if not hasattr(sympy, "_orig_primerange"):
    sympy._orig_primerange = sympy.primerange
    sympy.primerange = sympy.ntheory.primerange = patched_primerange

sympy.valuation = multiplicity
def _integer_valuation(self, p): return multiplicity(p, self)
sympy.Integer.valuation = _integer_valuation
sympy.core.numbers.Integer.valuation = _integer_valuation

def _return_self(self): return self
fractions.Fraction.reduced = _return_self
sympy.Rational.reduced = _return_self
sympy.Integer.reduced = _return_self

def _lcm(a: int, b: int) -> int:
    a = int(a); b = int(b)
    return 0 if a == 0 or b == 0 else a // math.gcd(a, b) * b

try:
    from sympy.ntheory.factor_ import carmichael as _carmichael
except Exception:
    _carmichael = None

if _carmichael is None:
    def carmichael(n: int) -> int:
        n = int(n)
        if n <= 0:
            raise ValueError("carmichael(n) requires n>0")
        fac = sp.factorint(n)
        lam = 1
        for p, e in fac.items():
            p = int(p); e = int(e)
            if p == 2 and e >= 3:
                # λ(2^e) = 2^(e-2) for e>=3
                pe = 1 << (e - 2)
            else:
                # λ(p^e) = φ(p^e) = p^(e-1)*(p-1) for odd p, and for p=2, e<=2
                pe = (p ** (e - 1)) * (p - 1)
            lam = _lcm(lam, pe)
        return int(lam)
else:
    carmichael = _carmichael

sympy.ntheory.carmichael = sympy.ntheory.residue_ntheory.carmichael = carmichael

def factorial_mod(n, m):
    res = 1
    for i in range(2, n + 1):
        res = (res * i) % m
    return res
sympy.factorial_mod = sympy.ntheory.factorial_mod = sympy.ntheory.modular.factorial_mod = factorial_mod
sympy.mod_factorial = sympy.ntheory.mod_factorial = sympy.ntheory.modular.mod_factorial = factorial_mod

def safe_head(obj, n=5):
    if isinstance(obj, dict):
        return list(obj.items())[:n]
    try:
        return obj[:n]
    except Exception:
        return obj

def safe_pow(base, exp, mod=None):
    if mod is not None:
        try:
            _b = int(base)
            _e = int(exp)
            _m = int(mod)
            if _m == 1: return 0
            if _m == 0: raise ZeroDivisionError("pow() modulus is 0")
            if _m < 0: raise ValueError("pow() negative modulus")
            return builtins._orig_pow(_b, _e, _m)
        except (TypeError, ValueError, AttributeError):
            return builtins._orig_pow(base, exp, mod)
    return builtins._orig_pow(base, exp)

if not hasattr(builtins, "_orig_pow"):
    builtins._orig_pow = builtins.pow
    builtins.pow = safe_pow

def _safe_repr(x):
    try:
        if isinstance(x, int):
            # If very large, do NOT materialize full decimal string
            bl = x.bit_length()
            if bl > BIT_THRESHOLD:
                return f"<int bit_length={{bl}}>"
        s = str(x)
    except Exception:
        s = repr(x)

    # Truncate giant strings
    if len(s) > MAX_STRSIZE:
        s = s[:10] + f"...<truncated len={{len(s)}}>" + s[-10:]
    return s

def safe_print(*args, **kwargs):
    # Keep print semantics but sanitize args
    sa = tuple(_safe_repr(a) for a in args)

    # Prevent huge container dumps (common failure mode)
    if len(sa) == 1 and isinstance(args[0], (list, tuple, dict, set)):
        sa = (_safe_repr(safe_head(args[0], 10)),)

    return builtins._orig_print(*sa, **kwargs)

if not hasattr(builtins, "_orig_print"):
    builtins._orig_print = builtins.print
    builtins.print = safe_print
""",
# Level 2: low risk monkey-patch, add new method to sympy
"""
def is_power_of_prime(n, p):
    n = int(n); p = int(p)
    if n <= 0 or p <= 1 or not sp.isprime(p):
        return False
    while n % p == 0:
        n //= p
    return n == 1

def discrete_root(a: int, n: int, m: int, *, all_roots: bool = False):
    a = int(a); n = int(n); m = int(m)
    if m <= 0 or n <= 0:
        raise ValueError("Require m>0 and n>0")
    a %= m
    if m == 1:
        return [0] if all_roots else 0
    if a == 0:
        # Keep simple: return 0 (a valid root). Not necessarily all roots.
        return [0] if all_roots else 0

    fac = sp.factorint(m)
    mod_list, roots_list = [], []

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
        roots_list.append([int(x) for x in r])

    if len(mod_list) == 1:
        sols = sorted({x % mod_list[0] for x in roots_list[0]})
        return sols if all_roots else sols[0]

    sols = set()
    for combo in product(*roots_list):
        x, _mod = crt(mod_list, combo)
        if x is not None:
            sols.add(int(x % m))

    sols = sorted(sols)
    return sols if all_roots else (sols[0] if sols else None)

sympy.isprimepower = is_power_of_prime
sympy.discrete_root = discrete_root
sympy.ntheory.residue_ntheory.discrete_root = discrete_root
""",
# Level 3: high risk global monkey-patch / cache / 安全限制改动
# 目标：只有当你明确需要时再开。这里包含你原来最可能引入“隐性漂移”的东西。
"""
import pickle
from joblib import Memory
from functools import wraps

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
""",
        ]

        self._helper_level = max(1, min(CFG.sandbox_helper_level, len(self._helper_codes) or 1))
        select_codes = self._helper_codes[:self._helper_level]
        if CFG.debug:
            CFG.sandbox_helper_codes = '\n'.join(select_codes)
        self.execute(select_codes)

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    @staticmethod
    def _truncate_text(s: str, *, max_chars: int, head: int, tail: int, max_lines: int) -> str:
        if not s:
            return s or ""
        # cap lines
        lines = s.splitlines()
        if len(lines) > max_lines:
            lines = lines[: (max_lines // 2)] + ["...[TRUNCATED LINES]..."] + lines[-(max_lines // 2):]
            s = "\n".join(lines)

        # cap chars
        if len(s) > max_chars:
            s = s[:head] + f"\n...[TRUNCATED CHARS total={len(s)}]...\n" + s[-tail:]
        return s

    def execute(self, code: str | Iterable[str], timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

        if isinstance(code, (list, tuple)):
            code = "\n\n".join(c.rstrip() for c in code if isinstance(c, str) and c.strip())

        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                # return f'[ERROR] Execution timed out after {effective_timeout} seconds'
                return f'{ERROR_TIMEOUT} {effective_timeout}s'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        # B) hard truncate to avoid poisoning later turns
        stdout = self._truncate_text(
            stdout,
            max_chars=CFG.tool_output_max_chars,
            head=CFG.tool_output_head,
            tail=CFG.tool_output_tail,
            max_lines=CFG.tool_output_max_lines,
        )
        stderr = self._truncate_text(
            stderr,
            max_chars=CFG.tool_output_max_chars,
            head=CFG.tool_output_head,
            tail=CFG.tool_output_tail,
            max_lines=CFG.tool_output_max_lines,
        )
        
        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def add_vars(self, name: str, vars: dict = {}):
        vars_repr = ", ".join([f"{k}={repr(v)}" for k, v in vars.items()])
        if not vars_repr:
            return
        execute_cmd = f"""
from types import SimpleNamespace
{name} = SimpleNamespace({vars_repr})
"""
        self.execute(f"{execute_cmd}")

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        codes = ['%reset -f\n']
        codes.extend(self._helper_codes[:self._helper_level])
        self.execute(codes)

    def __del__(self):
        self.close()

# timeout sanitize
def _tail_lines(s: str, n: int = 30) -> str:
    lines = (s or "").splitlines()
    k = len(lines)
    if k <= n: return s
    return "\n".join(lines[:2] + lines[-(n - 2):])

class AIMO3Tool:

    def __init__(self, cfg, sandbox=None):
        self.cfg = cfg
        self._local_jupyter_timeout = cfg.jupyter_timeout
        self._tool_prompt = cfg.tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, code: str) -> str:
        self._ensure_session()
        # final_script = self._ensure_last_print(code)
        with self._execution_lock:
            try:
                # output = self._jupyter_session.execute(code)
                output = self._jupyter_session.execute(code, timeout=self._local_jupyter_timeout)
            except TimeoutError as exc:
                output = f'{ERROR_TIMEOUT} {exc}'
        return output

class AIMO3Server:

    def __init__(self, cfg):
        self.cfg = cfg
        self.port = cfg.vllm_port
        self.base_url = cfg.vllm_base_url
        self.api_key = 'sk-local'
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
        self._preload_model_weights()
        self.server_process = self._start_server()
        self._wait_for_server()
    
    def _preload_model_weights(self) -> None:
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        files_to_load = []
        total_size = 0
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server', 
            '--seed', str(self.cfg.seed), 
            '--model', self.cfg.model_path, 
            '--served-model-name', self.cfg.served_model_name, 
            '--tensor-parallel-size', '1', 
            '--max-num-seqs', str(self.cfg.batch_size), 
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization), 
            '--host', '0.0.0.0', 
            '--port', str(self.port), 
            '--dtype', self.cfg.dtype, 
            '--kv-cache-dtype', self.cfg.kv_cache_dtype, 
            '--max-model-len', str(self.cfg.context_tokens), 
            '--stream-interval', str(self.cfg.stream_interval), 
            '--async-scheduling', 
            '--disable-log-stats', 
            '--enable-prefix-caching'
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )
    
    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        start_time = time.time()
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
            if return_code is not None:
                self.log_file.flush()
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
    
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError('Server failed to start (timeout).\n')

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'):
            self.log_file.close()

MAIN_CLASSES = ["NT", "ALG", "COMB", "GEO", "UNK"]
SUBCLASS_BY_MAIN: dict[str, list[str]] = {
    "NT": [
        "modular",                  # 同余、CRT、阶/周期的轻度也可先放 modular
        "mod-exponent",             # 大指数/阶/欧拉定理一类“指数模”题，和 modular 区分开更好写 discovery
        "diophantine",              # 丢番图方程、整数解/整除条件
        "gcd-lcm",                  # gcd/lcm、互素结构、整除链
        "prime-factorization",      # 质因数分解、素数、Omega/omega
        "valuation-lte",            # p-adic 估值、LTE、Legendre
        "orders-primitive-root",    # 阶、原根、指数循环，单列出来更稳
        "quadratic-residue",        # 二次剩余、Legendre/Jacobi
        "arithmetic-functions",     # phi/sigma/mu/tau 等、卷积
        "nt-sequence",              # 数论递推/构造序列
    ],
    "ALG": [
        "inequalities",             #
        "functional-equations",     #
        "polynomials",              # 
        "systems-equations",        # 方程组、代数消元
        "sequences-recursions",     # 代数递推、数列
        "symmetry-vieta",           # 对称多项式、Vieta、根与系数
        "complex",                  # 纯代数复数/单位根，不是几何复平面
        "floor-ceil",               # 整值/取整、分段
        "linear-algebra",           # 线代、矩阵、行列式
        "algebraic-identity",       # 恒等变形、配方、分解
    ],
    "COMB": [
        "counting",                 # 计数、容斥、递推计数
        "pigeonhole",
        "invariants-monovariants",
        "extremal",                 # 极值原理
        "graph",                    # 图论
        "probability",              # 概率/期望
        "combinatorial-game",       # 博弈/Nim/策略
        "generating-functions",
        "set-systems",              # 集合族、交并结构、Erdos–Ko–Rado 风格
        "combinatorial-geometry",   # 组合几何点线配置的“计数/抽屉”类，和 GEO 区分
        "recurrence",               # 偏组合的递推/DP/自动机
        "constructive",             # 构造/算法式证明
    ],
    "GEO": [
        "angle-chasing",            # 角追/圆周角
        "similarity",               # 相似/位似/螺旋相似
        "circle-power",             # 幂定理/根轴/根心
        "inversion",
        "projective",               # 射影/调和/梅涅劳斯/塞瓦扩展
        "coordinate",               # 坐标几何
        "vector",                   # 向量法
        "complex-geo",              # 复平面几何
        "trig-geo",                 # 三角法/正弦定理余弦定理驱动
        "transformations",          # 对称/旋转/平移/反演之外的变换
        "3d-geometry",
        "geo-inequality",           # 几何不等式单列，便于给不同 discovery
    ],
    "UNK": [
        "unknown",
    ],
}

# very small cross-type tags (optional)
CROSS_SUBCLASSES = [
    "mixed",            # 明显混合
    "hybrid-nt-comb",
    "hybrid-alg-nt",
    "hybrid-geo-alg",
]

def all_subclasses() -> list[str]:
    s = set()
    for v in SUBCLASS_BY_MAIN.values():
        s.update(v)
    s.update(CROSS_SUBCLASSES)
    # keep stable order for prompts
    return sorted(s)

ALL_SUBCLASSES = all_subclasses()

def allowed_subclasses_for_main(main_class: str) -> set[str]:
    main_class = (main_class or "UNK").upper()
    if main_class not in SUBCLASS_BY_MAIN:
        main_class = "UNK"
    return set(SUBCLASS_BY_MAIN[main_class]) | {"unknown"}  # always allow fallback

END_OF_CLASSIFICATION = "END_OF_CLASSIFICATION"
END_OF_PLAN = "END_OF_PLAN"
END_OF_DISCOVERY = "END_OF_DISCOVERY"

BOOST_PROMPTS: dict[str, str] = {
    "NT": (
        "NT BOOST (pick only what applies; keep concise):\n"
        "- Modular arithmetic / CRT; split prime powers.\n"
        "- Factorials/exponents: use valuations/LTE/Legendre when relevant.\n"
        "- Huge exponent B^(K) (esp. K multiple of many orders/phi): for many small d with gcd(d,B)=1, often B^K ≡ 1 (mod d).\n"
        "\n"
        "NT RECIPE: minimal integer from divisor-sum constraints (LCM-under-sum / divisor triple-sum / 'smallest N with divisors...'):\n"
        "- Sanity lower bound: if you need k distinct divisors summing to n and all divide X, then each ≤ X so n ≤ kX ⇒ X ≥ n/k.\n"
        "- Re-model: choose divisors a1<...<ak with sum n; feasibility requires ai | X; the minimal feasible X is typically lcm(a1,...,ak).\n"
        "- Therefore the core subproblem becomes: for a given n, minimize lcm(triple)/n (or lcm(triple) itself) over divisor tuples.\n"
        "- Do a SMALL Python discovery first (e.g. n≤200): enumerate tuples, compute best ratios r(n)=f(n)/n, keep top candidates with witnesses.\n"
        "- Scaling trick: if S|N and construction scales, often f(N) ≤ (N/S)·f(S) ⇒ f(N)/N ≤ f(S)/S. So scan only a shortlist of good S.\n"
        "- If N has form B^K + c with huge K and gcd(S,B)=1: if K multiple of ord_S(B) (or φ(S) when applicable), then B^K ≡ 1 (mod S)\n"
        "  ⇒ N ≡ 1+c (mod S), turning S|N into a small-divisor check on (c+1) (or similar simple residue).\n"
        "\n"
        "NORMALIZATION RULE (MANDATORY):\n"
        "- If the problem defines g(c)=f(N)/M but N=M+c, do NOT confuse f(N)/N with f(N)/M.\n"
        "- If you estimate r=f(N)/N, then f(N)/M = (N/M)·r. If c=t·M exactly, multiplier is (t+1).\n"
        "\n"
        "NT RED LINES (avoid common false shortcuts):\n"
        "- Do NOT use one-line heuristics (e.g. 'min prime factor of something') unless brute-verified on a small range.\n"
        "- Always distinguish base M vs evaluated N=M+c when normalizing.\n"
    ),
    "ALG": (
        "ALG BOOST (use only what applies; keep concise):\n"
        "- For inequalities: normalize, homogenize, try AM-GM/C-S/Jensen, tangent line.\n"
        "- For functional eq: plug special values, symmetry, injective/surjective, iterate.\n"
        "- For polynomials: factor patterns, Vieta, roots of unity filters.\n"
    ),
    "COMB": (
        "COMB BOOST (use only what applies; keep concise):\n"
        "- Invariants/monovariants and extremal principles first.\n"
        "- Counting: try double counting, bijection, recursion, inclusion-exclusion.\n"
        "- Graph: degrees, handshaking, connectivity, extremal bounds.\n"
    ),
    "GEO": (
        "GEO BOOST (use only what applies; keep concise):\n"
        "- Draw diagram mentally; chase angles, look for cyclic quadrilaterals.\n"
        "- Use power of a point, inversion, homothety/spiral similarity when stuck.\n"
        "- Coordinate/vector/complex only if it simplifies globally.\n"
    ),
    "UNK": (
        "GENERAL BOOST:\n"
        "- Restate the problem, identify structure, try small cases, then generalize.\n"
        "- Keep algebra clean; check edge cases; verify final answer.\n"
    ),
}

DISCOVERY_PROMPTS: dict[str, dict[str, str]] = {
    "NT": {
        "_default": (
            "NT DISCOVERY:\n"
            "- Identify modulus/prime factors; compute orders/valuations if relevant.\n"
            "- Try small cases to guess pattern; propose a lemma and prove it.\n"
            "- Decide which theorem applies (Euler/Fermat/CRT/LTE).\n"
            "- If the statement is about divisors / 'smallest integer with certain divisors' / divisor-sum constraints, "
            "switch to NT(arithmetic-functions) or NT(prime-factorization) mode.\n"
        ),
        "arithmetic-functions": (
            "NT(arithmetic-functions) DISCOVERY:\n"
            "- Identify the exact arithmetic object: divisor constraints, divisor sums, tau/sigma/phi, or 'smallest N with k distinct divisors meeting condition'.\n"
            "- If the task is 'exist k distinct divisors summing to n' and you want the minimal N:\n"
            "  treat it as minimizing L = lcm(divisors) subject to sum constraint; record witness tuples.\n"
            "\n"
            "Python micro-discovery template (ONLY if it helps; keep it small and structured):\n"
            "1) For n in a small range (e.g. 6..200), enumerate tuples a1<...<ak with sum=n (k given by problem; often 3),\n"
            "   compute L=lcm(tuple) and keep the best ratio r(n)=L/n (store as Fraction) with a witness tuple.\n"
            "   Sanity: enforce r(n) >= 1/k from n<=kL.\n"
            "2) Keep a shortlist of best n (top 20 by r(n) or by L/n) as reusable patterns.\n"
            "3) Per-instance: for each queried N, test whether some shortlisted pattern sum S is compatible (e.g. S|N or S|simple residue of N),\n"
            "   then scale the witness tuple if scaling is allowed by the construction.\n"
            "\n"
            "If N has a huge-exponent form B^K + c and gcd(S,B)=1, try reducing B^K mod S via ord/phi to make compatibility checks cheap.\n"
            "Stop once you have (i) a valid pattern family, (ii) a correctness argument (not just brute), and (iii) a clear aggregation plan.\n"
        ),
        "modular": (
            "NT(modular) DISCOVERY:\n"
            "- Factor modulus; handle each prime power; recombine via CRT.\n"
            "- Reduce exponents using order; consider gcd(base, mod) carefully.\n"
        ),
        "diophantine": (
            "NT(diophantine) DISCOVERY:\n"
            "- Search for modular obstructions and valuation constraints.\n"
            "- Try bounding (size/monotonicity), descent, or parameterization.\n"
        ),
        "prime-factorization": (
            "NT(prime-factorization) DISCOVERY:\n"
            "- Represent candidates via prime-exponent vectors; lcm corresponds to max exponents.\n"
            "- For 'minimize N' style problems, compare sizes by shifting mass to smaller primes and reducing exponents where possible.\n"
            "- If searching tuples of divisors, express each divisor as product of primes and reason about how their lcm constraints force N.\n"
            "- Use a SMALL brute/verification only to guess patterns, then prove via exponent inequalities / minimality exchange arguments.\n"
        ),
        "valuation-lte": (
            "NT(valuation-lte) DISCOVERY:\n"
            "- Compute v_p systematically; check LTE conditions; separate p=2.\n"
        ),
    },
    "ALG": {
        "_default": (
            "ALG DISCOVERY:\n"
            "- Identify whether inequality/FE/polynomial; list standard moves.\n"
            "- Try substitutions to simplify; look for symmetry and invariants.\n"
        ),
        "inequalities": (
            "ALG(inequalities) DISCOVERY:\n"
            "- Normalize/homogenize; try C-S/AM-GM/Jensen; look for equality cases.\n"
        ),
        "functional-equations": (
            "ALG(functional-equations) DISCOVERY:\n"
            "- Plug 0,1,-1; swap variables; derive injective/surjective; iterate.\n"
        ),
        "polynomials": (
            "ALG(polynomials) DISCOVERY:\n"
            "- Check factorization; use Vieta; try evaluating at special points/roots of unity.\n"
        ),
    },
    "COMB": {
        "_default": (
            "COMB DISCOVERY:\n"
            "- Decide: counting vs invariant vs extremal vs graph.\n"
            "- Try small n; look for bijection/double counting; define invariant if process.\n"
        ),
        "counting": (
            "COMB(counting) DISCOVERY:\n"
            "- Try recursion + base cases; double counting; inclusion-exclusion.\n"
        ),
        "invariants-monovariants": (
            "COMB(invariants) DISCOVERY:\n"
            "- Propose invariant/monovariant; prove it is preserved/decreases.\n"
        ),
        "graph": (
            "COMB(graph) DISCOVERY:\n"
            "- Degree constraints; extremal; consider BFS layers, cycles, matchings.\n"
        ),
        "generating-functions": (
            "COMB(gf) DISCOVERY:\n"
            "- Translate to polynomial/series; identify coefficient extraction target.\n"
        ),
    },
    "GEO": {
        "_default": (
            "GEO DISCOVERY:\n"
            "- Angle chase; look for cyclic points; try power of a point.\n"
            "- If messy: consider inversion or coordinates as last resort.\n"
        ),
        "circle-power": (
            "GEO(circle-power) DISCOVERY:\n"
            "- Identify radical axis/power relations; look for equal tangents.\n"
        ),
        "inversion": (
            "GEO(inversion) DISCOVERY:\n"
            "- Choose inversion center/radius to simplify circles/lines; track images.\n"
        ),
        "projective": (
            "GEO(projective) DISCOVERY:\n"
            "- Try Menelaus/Ceva, harmonic bundles, or projective transformations.\n"
        ),
    },
    "UNK": {
        "_default": (
            "DISCOVERY:\n"
            "- Try small cases; guess pattern; pick the most plausible main tool.\n"
        )
    }
}

def get_boost_prompt(main_class: str) -> str:
    main_class = (main_class or "UNK").upper()
    return BOOST_PROMPTS.get(main_class, BOOST_PROMPTS["UNK"])

def get_discovery_prompt(main_class: str, sub_class: str) -> str:
    main_class = (main_class or "UNK").upper()
    sub_class = (sub_class or "_default").strip()
    table = DISCOVERY_PROMPTS.get(main_class) or DISCOVERY_PROMPTS["UNK"]
    return table.get(sub_class, table.get("_default", DISCOVERY_PROMPTS["UNK"]["_default"]))

def get_subclass_by_main_lines() -> str:
    lines = []
    for k in MAIN_CLASSES:
        subs = SUBCLASS_BY_MAIN.get(k, [])
        lines.append(f"- {k}: " + ", ".join(subs))
    return "\n".join(lines)

SUBCLASS_BY_MAIN_LINES = get_subclass_by_main_lines()

TURN0_CLASSIFY_SYS = (
    "You are a strict math-problem classifier.\n"
    "Output EXACTLY 4 lines, no extra text and respond in the final channel ONLY:\n"
    "Main: <NT|ALG|COMB|GEO|UNK>\n"
    "Sub: <one sub class from the allowed list for that Main>\n"
    "Hybrid: <NONE|NT|ALG|COMB|GEO|e.g. COMB-NT>\n"
    f"{END_OF_CLASSIFICATION}\n"

    "\nRules:\n"
    "- Main must be one of NT|ALG|COMB|GEO|UNK.\n"
    "- Sub must be a SINGLE token from the list under Main.\n"
    "- Use Hybrid only if clearly cross-type; otherwise Hybrid: NONE.\n"

    "\nAllowed Sub classes by Main (Sub must match exactly):\n"
    f"{SUBCLASS_BY_MAIN_LINES}\n"

    '\nNT subclass guidelines (pick the MOST specific):\n'
    '- arithmetic-functions: problems about divisors/functions like tau/sigma/phi/mu, divisor sums, "smallest N with a divisor property",'
    'constructing minimal integer given constraints on distinct divisors or divisor-sum conditions.\n'
    '- prime-factorization: reasoning mainly by prime exponent vectors; "minimize N" by choosing prime powers; compare sizes via exponents.\n'
    '- diophantine: explicitly solving integer equations/tuples, integer solutions, parametric forms, descent (not merely divisor construction).\n'
    '- modular / mod-exponent / valuation-lte: congruences/orders/LTE/valuations are the main engine.\n\n'
    '**Rule**: if the statement emphasizes "divisors", "three distinct divisors", "sum of divisors", "smallest integer with such divisors",'
    'prefer arithmetic-functions (or prime-factorization if optimization over prime exponents dominates), NOT diophantine.\n'
)

TURN1_PLAN_SYS = (
    "You are a planner. Produce a concise plan only in the final channel.\n"
    "Output format:\n"
    "Plan:\n"
    "- ...\n"
    "- ...\n"
    f"{END_OF_PLAN}\n\n"
    "Rules:\n"
    "- 3~7 bullets, each bullet <= 120 chars.\n"
    "- No proof, no full solution, no final answer.\n"
)

@dataclass
class AttemptState:
    main_class: str = "UNK"
    sub_class: str = "unknown"
    hybrid: str = "NONE"
    plan: str = ""
    classification_raw: str = ""
    class_conf = 0.0
    plan_raw: str = ""
    discovery_raw: str = ""
    boost_inject: bool = False
    
    problem_id: str = ""
    attempt_index: int = 0
    seed: int = 0
    turn: int = 0
    python_calls: int = 0
    python_errors: int = 0
    timeout_errors: int = 0
    total_tokens: int = 0
    max_tokens: int = 0
    py_num_counter: dict = field(default_factory=dict)

@dataclass
class TurnCfg:
    name: str
    enabled: bool = True
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 1.0
    min_p: float = 0.02
    max_tokens: int = 2048
    stop_strings: Optional[list[str]] = None
    prefill: str = ""

    # runtime slots (filled after classification)
    main: str = ""
    sub: str = ""
    hybrid: str = ""

class RAG:

    discovery_sys_prompt = (
        "You are in the discovery phase.\n"
        "Goal: explore key lemmas/cases, propose useful transforms, and start reasoning.\n"
        f"Keep it structured and stop when done by writing {END_OF_DISCOVERY} on its own line.\n"
    )

    pre_turns_enabled = True
    turn0 = TurnCfg(
        name="t0_classify",
        temperature=0.0,
        min_p=0.0,
        max_tokens=100,
        stop_strings=[END_OF_CLASSIFICATION],
            # , "<|start|>assistant", "assistantanalysis", "assistant to=python"
            # , "```", "\nDefinition:", "\nObservation:", "\n###"],
        prefill="Main: ",
        system_prompt=TURN0_CLASSIFY_SYS,
    )
    turn1 = TurnCfg(
        name="t1_plan",
        temperature=0.2,
        min_p=0.0,
        max_tokens=250,
        stop_strings=[END_OF_PLAN],
        prefill="Plan:\n- ",
        system_prompt=TURN1_PLAN_SYS,
    )

    # TODO:
    rag_enabled = False
    rag_topK = 5

def _strip_after_sentinel(text: str, sentinel: str) -> str:
    if not text:
        return ""
    idx = text.find(sentinel)
    if idx >= 0:
        return text[:idx].rstrip()
    return text.rstrip()

def parse_classification(
    text: str,
    main_classes: list[str],
    allowed_subclasses_by_main: Callable[[str], set[str]],
) -> tuple[str, str, str]:
    raw = (text or "").strip()
    raw = _strip_after_sentinel(raw, "END_OF_CLASSIFICATION")

    # 清理常见粘连/标记（不追求完美，但要便宜）
    raw = re.sub(r"<\|.*?\|>", " ", raw)  # chatml tokens
    raw = re.sub(r"assistant(?:analysis|final)\s*", "", raw, flags=re.I)

    def _last(pattern: str) -> str | None:
        ms = re.findall(pattern, raw, flags=re.I)
        return ms[-1].strip() if ms else None

    # 允许 "assistantfinalMain: NT" 这种：不要求行首，直接 anywhere match
    main = (_last(r"main\s*:\s*([A-Za-z]{2,4})") or "UNK").upper()
    sub  = (_last(r"sub\s*:\s*([a-z][a-z\-]+)") or "unknown").lower()
    hy0  = (_last(r"hybrid\s*:\s*([A-Za-z0-9\-]+)") or "NONE").upper()

    # normalize main
    if main not in main_classes:
        main = "UNK"

    # normalize sub
    sub = re.sub(r"[^a-z\-]", "", sub)
    allowed = allowed_subclasses_by_main(main) if callable(allowed_subclasses_by_main) else set()
    if allowed and sub not in allowed:
        sub = "unknown"

    # normalize hybrid
    def _norm_hybrid(h: str) -> str:
        h = (h or "").strip().upper()
        if h in ("", "NO", "NONE", "FALSE"):
            return "NONE"
        # accept: HYBRID-NT-COMB / NT-COMB / COMB-NT
        h = re.sub(r"^HYBRID[-_:]*", "", h)
        parts = [p for p in re.split(r"[^A-Z]+", h) if p]
        parts = [p for p in parts if p in main_classes and p != "UNK"]
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "NONE"

    hybrid = _norm_hybrid(hy0)
    return main, sub, hybrid

def refine_nt_subclass(problem_text: str, sub: str) -> str:
    t = (problem_text or "").lower()

    if ("divisor" in t or "factor" in t or "约数" in t) and (
        "smallest" in t or "minimum" in t or "最小" in t
    ):
        if ("sum of" in t and "divisor" in t) or ("distinct" in t and ("divisor" in t or "factor" in t)):
            return "arithmetic-functions"
    if ("prime" in t or "素数" in t) and ("exponent" in t or "power" in t or "^" in t or "幂" in t):
        if ("min" in t or "smallest" in t or "minimum" in t or "最小" in t):
            return "prime-factorization"
    return sub

# can be used as supplementary to turn0's main/sub/hybrid
def robust_classifier(text: str) -> tuple[str, str, str]:
    """
    基于规则、关键词权重和逻辑特征的数学题目分类器。
    返回 (main_class, sub_class, hybrid)

    main_class ∈ MAIN_CLASSES
    sub_class  ∈ SUBCLASS_BY_MAIN[main_class]
    hybrid     ∈ CROSS_SUBCLASSES 或 ""（表示非混合）
    """
    if not text:
        return ("UNK", "unknown", "NONE")

    raw = text
    raw = re.sub(r"<\|.*?\|>", " ", raw)
    raw = re.sub(r"assistant(?:analysis|final)\s*", "", raw, flags=re.I)
    text = raw.lower()

    # -------------------------
    # 0) Hard rules (main + sub)
    # -------------------------
    hard_rules = [
        # NT
        (r"chinese remainder theorem|\bcrt\b", ("NT", "modular")),
        (r"euler's totient|\bphi\b|\btotient\b", ("NT", "arithmetic-functions")),
        (r"fermat's little theorem", ("NT", "mod-exponent")),
        (r"egyptian fraction|diophantine|\binteger solutions\b|\bsolve in integers\b", ("NT", "diophantine")),

        # GEO
        (r"incircle|circumcircle|orthocenter|concyclic", ("GEO", "angle-chasing")),
        (r"sine rule|cosine rule|\btrigonometric\b", ("GEO", "trig-geo")),
        (r"similarity of triangles|\bsimilar\b", ("GEO", "similarity")),

        # COMB
        (r"pigeonhole principle", ("COMB", "pigeonhole")),
        (r"inclusion-exclusion", ("COMB", "counting")),
        (r"generating function", ("COMB", "generating-functions")),
    ]

    for pat, (m, s) in hard_rules:
        if re.search(pat, text):
            return (m, s, "")

    # -------------------------
    # 1) Main class scoring
    # -------------------------
    scores = {"NT": 0, "ALG": 0, "COMB": 0, "GEO": 0}

    keywords = {
        "NT": {
            "divisor": 3, "prime": 3, "modulo": 3, "gcd": 2, "lcm": 2,
            "integer": 1, "congruent": 3, "factorization": 2, "divisibility": 3,
            "natural number": 1, "positive integer": 1, "residue": 2,
        },
        "ALG": {
            "polynomial": 3, "inequality": 3, "real number": 1, "function": 1,
            "coefficient": 2, "roots": 2, "quadratic": 2, "derivative": 2,
            "am-gm": 4, "cauchy-schwarz": 4, "sequence": 1, "equation": 1,
        },
        "COMB": {
            "arrangement": 3, "permutation": 3, "combination": 3, "probability": 3,
            "subset": 2, "grid": 2, "graph": 2, "vertex": 1, "edge": 1,
            "tournament": 3, "coloring": 3, "ways to": 2, "partition": 2,
        },
        "GEO": {
            "triangle": 2, "circle": 3, "angle": 2, "radius": 2, "parallel": 2,
            "perpendicular": 2, "area": 1, "length": 1, "coordinate": 2,
            "tangent": 3, "quadrilateral": 3, "polygon": 2,
        },
    }

    for cat, word_map in keywords.items():
        for word, weight in word_map.items():
            pat = r"\b" + re.escape(word) + r"\b"
            if re.search(pat, text):
                scores[cat] += weight

    # logic-feature compensation
    if re.search(r"\\pmod|\\equiv", text):
        scores["NT"] += 5
    if re.search(r"f\(x\)|p\(x\)|q\(x\)", text):
        scores["ALG"] += 3
    if re.search(r"\\sum_{i=1}\^{n}|\\binom\{n\}\{k\}", text):
        scores["COMB"] += 2
    if re.search(r"\\triangle|\\angle|\\pi", text):
        scores["GEO"] += 3
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bdetermine the number\b", text):
        scores["COMB"] += 4
    if re.search(r"\bthere exists\b|\bexists\b", text):
        scores["COMB"] += 1
        scores["ALG"] += 1

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    if best_score < 2:
        return ("UNK", "unknown", "")

    # -------------------------
    # 2) Hybrid tag (optional)
    # -------------------------
    sorted_cats = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    second_cat, second_score = sorted_cats[1]

    hybrid = ""
    # “接近且都不低”才判混合，避免噪声
    if second_score >= 3 and (best_score - second_score) <= 1:
        pair = {best_cat, second_cat}
        if pair == {"NT", "COMB"}:
            hybrid = "hybrid-nt-comb"
        elif pair == {"ALG", "NT"}:
            hybrid = "hybrid-alg-nt"
        elif pair == {"GEO", "ALG"}:
            hybrid = "hybrid-geo-alg"
        else:
            hybrid = "mixed"

    # -------------------------
    # 3) Subclass inference (within main)
    # -------------------------
    def _infer_subclass(main: str, t: str) -> str:
        if main == "UNK":
            return "unknown"

        # 每个 subclass 给一个轻量 signals；命中加分，最后取最高
        # （如果都不命中，fallback 到该 main 的第一个 subclass，保证返回值合法）
        sub_scores = {s: 0 for s in SUBCLASS_BY_MAIN[main]}

        def bump(sub: str, pat: str, w: int = 1):
            if sub in sub_scores and re.search(pat, t):
                sub_scores[sub] += w

        if main == "NT":
            bump("gcd-lcm", r"\bgcd\b|\blcm\b|\bcoprime\b|\brelatively prime\b", 4)
            bump("modular", r"\\pmod|\bmod\b|\bmodulo\b|\bcongruen", 4)
            bump("mod-exponent", r"\bmod\b|\bmodulo\b|\\pmod", 2)
            bump("mod-exponent", r"\^|!\b|\bpower\b|\bexponent\b", 3)
            bump("diophantine", r"\bdiophantine\b|\bintegers?\b.*\bsolution\b|\bsolve in integers\b", 5)
            bump("prime-factorization", r"\bprime\b|\bfactoriz|\bdivisor(s)?\b|\bcomposite\b", 3)
            bump("valuation-lte", r"\bvaluation\b|\bv_p\b|\bp-adic\b|\blte\b|\blegendre\b", 6)
            bump("orders-primitive-root", r"\border\b|\bord\b|\bprimitive root\b|\bgenerator\b", 6)
            bump("quadratic-residue", r"\bquadratic residue\b|\blegendre\b|\bjacobi\b|x\^2|\\left\(\s*\\frac", 6)
            bump("arithmetic-functions", r"\bphi\b|\btotient\b|\bsigma\b|\btau\b|\bmu\b|\bmobius\b", 6)
            bump("nt-sequence", r"\bsequence\b|a_n|\\{a_n\\}|\brecurrence\b", 2)

        elif main == "ALG":
            bump("inequalities", r"\binequality\b|\\ge|\\le|>=|<=|\bam-gm\b|\bcauchy\b|\bholder\b|\bjensen\b", 6)
            bump("functional-equations", r"f\(\s*x|\bfunctional equation\b|\bfor all\b|\bforall\b", 6)
            bump("polynomials", r"\bpolynomial\b|\bdegree\b|\bcoefficient\b|\broots?\b", 5)
            bump("symmetry-vieta", r"\bvieta\b|\bsymmetric\b|\bsum of roots\b|\bproduct of roots\b", 6)
            bump("systems-equations", r"\bsystem\b|\bsimultaneous\b|\bsolve\b.*\b( and |, )", 4)
            bump("sequences-recursions", r"\bsequence\b|a_n|\\{a_n\\}|\brecurrence\b|\brecursive\b", 4)
            bump("complex", r"\bcomplex\b|\bimaginary\b|\bre\b|\bim\b", 4)
            bump("floor-ceil", r"\\lfloor|\\lceil|\bfloor\b|\bceil\b|\bgreatest integer\b|\bfractional part\b", 6)
            bump("linear-algebra", r"\bmatrix\b|\bdeterminant\b|\beigen", 6)
            bump("algebraic-identity", r"\bidentity\b|\bfactor\b|\bexpand\b|\bsimplify\b|\bcomplete the square\b", 3)

        elif main == "COMB":
            bump("counting", r"\bhow many\b|\bnumber of\b|\bcount\b|\bways\b|\bpermutation\b|\bcombination\b", 6)
            bump("pigeonhole", r"\bpigeonhole\b|\bdirichlet\b", 7)
            bump("invariants-monovariants", r"\binvariant\b|\bmonovariant\b|\bparity\b", 6)
            bump("extremal", r"\bextremal\b|\bminimum\b|\bmaximum\b|\bminimize\b|\bmaximize\b", 5)
            bump("graph", r"\bgraph\b|\bvertex\b|\bedge\b|\bpath\b|\bcycle\b", 6)
            bump("probability", r"\bprobability\b|\bexpected\b|\brandom\b|\bvariance\b", 6)
            bump("combinatorial-game", r"\bgame\b|\bplayer\b|\bwinning strategy\b|\bnim\b", 6)
            bump("generating-functions", r"\bgenerating function\b", 7)
            bump("set-systems", r"\bfamily of sets\b|\bset system\b|\bintersection\b|\bunion\b|\bsubsets?\b", 5)
            bump("combinatorial-geometry", r"\bpoints?\b|\blines?\b|\bplane\b", 2)
            bump("recurrence", r"\brecurrence\b|\bdp\b|\bdynamic programming\b|\brecursive\b", 4)
            bump("constructive", r"\bconstruct\b|\balgorithm\b|\bexplicit\b|\bthere exists\b", 3)

        elif main == "GEO":
            bump("angle-chasing", r"\bangle\b|\\angle|\bconcyclic\b|\bcircumcircle\b|\bincircle\b|\borthocenter\b", 6)
            bump("similarity", r"\bsimilar\b|\bhomothety\b|\bratio\b", 6)
            bump("circle-power", r"\bpower of a point\b|\bradical axis\b|\bradical center\b", 7)
            bump("inversion", r"\binversion\b", 8)
            bump("projective", r"\bmenelaus\b|\bceva\b|\bharmonic\b|\bcross ratio\b|\bprojective\b", 7)
            bump("coordinate", r"\bcoordinate\b|\bcoordinates\b|\b(x\s*,\s*y)\b|x\s*=\s*|y\s*=\s*", 6)
            bump("vector", r"\bvector\b|\bdot product\b|\bcross product\b", 6)
            bump("complex-geo", r"\bcomplex plane\b|\baffix\b|\barg\b", 6)
            bump("trig-geo", r"\bsine\b|\bcosine\b|\btangent\b|\bsine rule\b|\bcosine rule\b", 7)
            bump("transformations", r"\brotation\b|\breflection\b|\btranslation\b|\bsymmetry\b", 5)
            bump("3d-geometry", r"\bsphere\b|\btetrahedron\b|\bcube\b|\bpolyhedron\b|\b3d\b", 7)
            bump("geo-inequality", r"\binequality\b|\\ge|\\le|>=|<=|\bmaximize\b|\bminimize\b", 3)

        best_sub = max(sub_scores, key=sub_scores.get)
        if sub_scores[best_sub] <= 0:
            # 保底：保证 sub_class 一定在 SUBCLASS_BY_MAIN[main] 里
            return SUBCLASS_BY_MAIN[main][0]
        return best_sub

    sub_class = _infer_subclass(best_cat, text)
    return (best_cat, sub_class, hybrid)

###### entropy computer ######
def _compute_mean_entropy_v1(logprobs: list, *, last_frac: float = 0.2, tail_boost: float = 2.0) -> float:
    if not logprobs:
        return float('inf')

    total_entropy = 0.0
    token_count = 0
    for top_logprobs_dict in logprobs:            
        if not isinstance(top_logprobs_dict, dict):
            continue
        if not top_logprobs_dict:
            continue
        token_entropy = 0.0
        
        for token_str, log_prob in top_logprobs_dict.items():
            prob = math.exp(log_prob)
            if prob > 0:
                token_entropy -= prob * math.log2(prob)
        
        total_entropy += token_entropy
        token_count += 1

    if token_count == 0:
        return float('inf')

    return total_entropy / token_count

def _compute_mean_entropy_v1d5(logprobs: list, *, last_frac: float = 0.2, tail_boost: float = 2.0) -> float:
    """
    v1.5: stable + useful for reranking
    - Compute per-token Shannon entropy from top_logprobs (approx).
    - Use a mild position emphasis: last `last_frac` tokens get `tail_boost` weight.
    - No hard thresholds, no streak logic (more robust across prompts/temps/top_logprobs).
    """
    if not logprobs:
        return float("inf")

    entropies = []
    for top_lp in logprobs:
        if not isinstance(top_lp, dict) or not top_lp:
            continue
        h = 0.0
        for lp in top_lp.values():
            # lp is log(prob)
            p = math.exp(lp)
            if p > 0:
                h -= p * math.log2(p)
        entropies.append(h)

    n = len(entropies)
    if n == 0:
        return float("inf")

    # tail window (at least 1 token)
    tail_n = max(1, int(n * last_frac))
    split = n - tail_n

    # weighted mean: head weight=1, tail weight=tail_boost
    head_sum = sum(entropies[:split])
    tail_sum = sum(entropies[split:])
    head_w = split * 1.0
    tail_w = tail_n * float(tail_boost)

    return (head_sum + tail_boost * tail_sum) / (head_w + tail_w)

MEAN_ENTROPY_COMPUTERS = {
    'v1': _compute_mean_entropy_v1,
    'v1d5': _compute_mean_entropy_v1d5,
}

def compute_mean_entropy(ver: str, logprobs_buffer: list, *, last_frac: float = 0.2, tail_boost: float = 2.0) -> float:
    if not logprobs_buffer or not ver or not isinstance(ver, str):
        return float('inf')
    
    fn = MEAN_ENTROPY_COMPUTERS.get(ver)
    if fn is None or not callable(fn):
        return float('inf')
    
    return fn(logprobs_buffer, last_frac=last_frac, tail_boost=tail_boost)

class AIMO3Solver:

    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.base_url = self.cfg.vllm_base_url
        self.api_key = 'sk-local'
        # timeout=httpx.Timeout(
        #     timeout=self.cfg.session_timeout,  # overall cap (可选)
        #     connect=5.0,
        #     read=self.cfg.session_timeout,     # 重点：流式时 read 太小会误杀
        #     write=30.0,
        #     pool=5.0,
        # ),
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        
        self.notebook_start_time = time.time()
        self.problems_remaining = 50
        self.logger = AIMO3Logger(CFG, RAG)
        self._initialize_kernels()
        self._test_lock = threading.Lock()

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')

    def _scan_for_answer(self, text: str) -> int | None:
        if not text:
            return None

        t = text.replace("assistantfinal", "").strip()

        patterns = [
            r'\\boxed\s*\{\D*([0-9](?:[0-9,]*[0-9])?)\D*\}',
            r'final\s+answer\s+is\s*([0-9,]+)',
            r'\bAnswer\s*[:=]\s*\**\s*([0-9,]+)\b',
            r'\*\*Answer:\s*([0-9,]+)\*\*',
            r'\banswer\s+is\s*([0-9,]+)\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, t, re.IGNORECASE)
            if matches:
                try:
                    clean_value = matches[-1].replace(',', '')
                    value = int(clean_value)
                    return value if 0 <= value <= 99999 else value % 99999
                except ValueError:
                    pass

        # 兜底：final message 只有一个数字
        if t.isdigit():
            value = int(t)
            return value if 0 <= value <= 99999 else value % 99999

        return None

    def _debug_req(self, attempt_log: deque, attempt_index: int, turn: int, prompt_ids: List[int]):
        if not self.cfg.debug or not self.cfg.debug_req:
            return

        full_request_text = self.encoding.decode(prompt_ids)
        snippet = self.logger.get_debug_snippet(full_request_text)
        # for Turn 0, 20,... log full request
        formatted_req = _format_markdown(full_request_text if turn % self.cfg.debug_req_full == 0 else snippet)
        attempt_log.append(f"### Turn {attempt_index + 1}.{turn} - Raw Request to Model:")
        attempt_log.append(formatted_req)

    def _debug_resp(self, attempt_log: deque, attempt_index: int, turn: int, full_response_text: str):
        if not self.cfg.debug or not self.cfg.debug_resp or not full_response_text:
            return

        attempt_log.append(f"### Turn {attempt_index + 1}.{turn} - Model Response:")
        formatted_resp = _format_markdown(full_response_text)
        attempt_log.append(formatted_resp)

    def _debug_python(self, attempt_log: deque, attempt_index: int, turn: int, 
                      raw_script: str, final_script: str, tool_output: str, sanitized_output: str, has_error: bool):
        if not self.cfg.debug:
            return

        emoji_error = '❌' if has_error else ''
        attempt_log.append(f"### Turn {attempt_index + 1}.{turn} - Python Raw:")
        attempt_log.append(_format_markdown(raw_script, mode='python'))
        attempt_log.append(f"### Turn {attempt_index + 1}.{turn} - Python Call:")
        attempt_log.append(_format_markdown(final_script, mode='python'))
        attempt_log.append(f"### Turn {attempt_index + 1}.{turn} {emoji_error} - Python Output:")
        snippet_out = self.logger.get_debug_snippet(tool_output)
        formatted_out = _format_markdown(snippet_out, mode='text')
        attempt_log.append(f"{formatted_out}\n")
        if tool_output != sanitized_output:
            attempt_log.append(f"### Turn {attempt_index + 1}.{turn} {emoji_error} - Sanitized Output:")
            snippet_out = self.logger.get_debug_snippet(sanitized_output)
            formatted_out = _format_markdown(snippet_out, mode='text')
            attempt_log.append(f"{formatted_out}\n")

    def _create_conversation(self, system_prompt, user_prompt, *, tool: ToolNamespaceConfig = None, 
                             developer_prompt: str = None, reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH):
        system_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=reasoning_effort)
        )
        if tool is not None: 
            system_content.with_tools(tool)

        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, user_prompt), # problem
        ]
        if developer_prompt is not None:
            messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_prompt))
        return Conversation.from_messages(messages)

    def _run_preturn(self, problem: str, tc: TurnCfg, state: AttemptState,
                    deadline: float, stop_event: threading.Event, 
                    attempt_log: deque, reasoning_effort: ReasoningEffort = ReasoningEffort.LOW):
        if not tc.enabled:
            return ""

        temp_conv = self._create_conversation(tc.system_prompt, problem, reasoning_effort = reasoning_effort)
        prompt_ids = self.encoding.render_conversation_for_completion(temp_conv, Role.ASSISTANT)

        # prefill 仍然放在 prompt 里（维持你现在的做法）
        if tc.prefill:
            prompt_ids = prompt_ids + self.encoding.encode(tc.prefill)

        if self.cfg.debug and self.cfg.debug_req:
            request_text = self.encoding.decode(prompt_ids)
            attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Raw Request to Model:")
            attempt_log.append(_format_markdown(request_text))

        stream = self.client.completions.create(
            model=tc.model_name or self.cfg.served_model_name,
            temperature=float(tc.temperature),
            max_tokens=int(tc.max_tokens),
            prompt=prompt_ids,
            seed=state.seed,
            stream=True,
            stop=(tc.stop_strings or None),
            extra_body={
                "min_p": float(tc.min_p),
                "stop_token_ids": self.stop_token_ids,
                "return_token_ids": True,
            },
        )

        token_buffer: list[int] = []
        text_chunks: list[str] = []
        try:
            for chunk in stream:
                if stop_event.is_set() or time.time() > deadline:
                    break
                new_tokens = getattr(chunk.choices[0], "token_ids", None)
                if new_tokens:
                    token_buffer.extend(new_tokens)
                    state.total_tokens += len(new_tokens)

                new_text = getattr(chunk.choices[0], "text", "") or ""
                if new_text:
                    text_chunks.append(new_text)
        finally:
            stream.close()

        raw_text = "".join(text_chunks).strip()
        if not token_buffer and not raw_text:
            return ""

        # Prefer structured parse, but NEVER fail the preturn if it breaks
        out = ""
        if token_buffer:
            try:
                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)

                final_text = None
                last_text = ""
                for m in new_messages:
                    txt = ""
                    try:
                        if m.content and hasattr(m.content[0], "text"):
                            txt = m.content[0].text or ""
                    except Exception:
                        txt = ""
                    if txt:
                        last_text = txt
                    if getattr(m, "channel", None) == "final" and txt:
                        final_text = txt

                out = (final_text or last_text or "").strip()
            except Exception as e:
                # downgrade to debug note; fallback to raw text
                if self.cfg.debug:
                    attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Preturn Parse Fallback:")
                    attempt_log.append(_format_markdown(str(e), mode="text"))
                out = raw_text
        else:
            out = raw_text

        # If we used prefill, prepend it back when missing (prefill is in prompt, not generation)
        if tc.prefill:
            pref = tc.prefill.strip()
            if pref and not out.lstrip().startswith(pref):
                out = tc.prefill + out

        out = out.strip()

        if self.cfg.debug and self.cfg.debug_resp:
            attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Model Response:")
            attempt_log.append(_format_markdown(out))

        return out

    def _run_turn0(self, problem: str, tc: TurnCfg, state: AttemptState, deadline: float, stop_event: threading.Event, attempt_log: deque):
        if not tc.enabled: return
        state.turn = -2
        out = self._run_preturn(problem, tc, state, deadline, stop_event, attempt_log, ReasoningEffort.MEDIUM)
        state.classification_raw = _strip_after_sentinel(out or "", END_OF_CLASSIFICATION)

        m1, s1, h1 = parse_classification(state.classification_raw, MAIN_CLASSES, allowed_subclasses_for_main)
        m2, s2, h2 = robust_classifier(problem)
        if m1 == "NT":
            s1 = refine_nt_subclass(problem, s1)

        if m1 != "UNK" and s1 != "unknown":
            state.main_class, state.sub_class, state.hybrid = m1, s1, h1
            state.class_conf = 0.9 if (m1 == m2 and s1 == s2) else 0.6
        elif s1 == "unknown" and s2 != "unknown":
            state.sub_class = s2
            state.class_conf = 0.6
        else:
            state.main_class, state.sub_class, state.hybrid = m2, s2, h2
            state.class_conf = 0.5

        if self.cfg.debug:
            attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Classification:")
            note = f"({m1},{s1},{h1}) ({m2},{s2},{h2}) => ({state.main_class},{state.sub_class},{state.hybrid},{state.class_conf:.2f})"
            attempt_log.append(note)
            # print(f"classification_raw: {state.classification_raw}")
            print(note)

    def _run_turn1(self, problem: str, tc: TurnCfg, state: AttemptState, deadline: float, stop_event: threading.Event, attempt_log: deque):
        if not tc.enabled: return
        state.turn = -1
        state.plan_raw = self._run_preturn(problem, tc, state, deadline, stop_event, attempt_log, ReasoningEffort.MEDIUM)
        state.plan = _strip_after_sentinel(state.plan_raw or "", END_OF_PLAN)

        if self.cfg.debug and state.plan:
            attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Plan:")
            attempt_log.append(_format_markdown(state.plan, 'quote'))
            print(f"{state.problem_id}: Turn {state.attempt_index + 1}.{state.turn} => Plan({len(state.plan)}):\n{state.plan}")

    def _guide(self, problem: str, state: AttemptState, conv: Conversation, attempt_log: deque):
        if state.boost_inject:
            return

        if state.plan:
            conv.messages.append(Message.from_role_and_content(Role.ASSISTANT, state.plan))

        boost_prompt = get_boost_prompt(state.main_class)
        if boost_prompt:                
            conv.messages.append(Message.from_role_and_content(Role.DEVELOPER, boost_prompt))

        discovery_prompt = get_discovery_prompt(state.main_class, state.sub_class)
        if discovery_prompt:
            conv.messages.append(Message.from_role_and_content(Role.DEVELOPER, discovery_prompt))

        state.boost_inject = True

        if self.cfg.debug and (boost_prompt or discovery_prompt):
            attempt_log.append(f"### Turn {state.attempt_index + 1}.{state.turn} - Cheat Sheet:")
            attempt_log.append(f"**boost_prompt**:\n{_format_markdown(boost_prompt, 'quote')}")
            attempt_log.append(f"**discovery_prompt**:\n{_format_markdown(discovery_prompt, 'quote')}")

    def _update_counter(self, output: str, counter: dict = {}):
        # counter
        try:
            s = (output or "").strip()
            # Only count if output is exactly ONE integer and within answer range
            if len(s) <= 32 and re.fullmatch(r"[+-]?\d{1,5}", s):
                n = int(s)
                if 0 <= n <= 99999:
                    counter[n] = counter.get(n, 0) + 1
        except Exception:
            pass

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float,
        problem_id: str,
    ) -> dict:
        attempt_log = deque([])
        attempt_start = time.time()

        if stop_event.is_set() or time.time() > deadline:
            return {
                'Attempt': attempt_index + 1, 
                'Answer': None, 
                'Python Calls': 0, 
                'Python Errors': 0, 
                'Timeout Errors': 0,
                'Response Length': 0, 
                'Entropy': float('inf'),
                'Time': '0:00',
                'TPS': 0, # token per second, to indicate vllm/model's healthy
                '_Log': "\n".join(attempt_log),
                '_Elapsed': 0,
            }
    
        local_tool = None
        sandbox = None
        final_answer = None
        logprobs_buffer = []
        state = AttemptState()
        state.problem_id = problem_id
        state.attempt_index = attempt_index
        state.seed = (self.cfg.seed + attempt_index) ** 2

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)  
            local_tool = AIMO3Tool(self.cfg, sandbox=sandbox)
            # sandbox.add_vars("CTX_ATT", {"pid": problem_id, "idx": attempt_index + 1})
            # self._test_sandbox(sandbox)

            encoding = self.encoding
            user_input = f'{problem}\n\n{self.cfg.preference_prompt}'
            conversation = self._create_conversation(system_prompt, user_input, tool=local_tool.tool_config, reasoning_effort=ReasoningEffort.HIGH)
            if RAG.pre_turns_enabled:
                self._run_turn0(problem, RAG.turn0, state, deadline, stop_event, attempt_log)
                self._run_turn1(problem, RAG.turn1, state, deadline, stop_event, attempt_log)
                self._guide(problem, state, conversation, attempt_log)

            for turn_i in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                state.turn = turn_i
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
    
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, 
                    temperature=self.cfg.temperature, 
                    logprobs=self.cfg.top_logprobs, 
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=state.seed, 
                    stream=True, 
                    extra_body={
                        'min_p': self.cfg.min_p, 
                        'stop_token_ids': self.stop_token_ids, 
                        'return_token_ids': True
                    }
                )

                self._debug_req(attempt_log, attempt_index, turn_i, prompt_ids)
    
                full_response_text = ""
                try:
                    token_buffer = []
                    text_chunks = []
    
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
    
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
    
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            state.total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            if self.cfg.debug and self.cfg.debug_resp:
                                full_response_text += new_text

                            chunk_logprobs = chunk.choices[0].logprobs
                            if chunk_logprobs is not None:
                                if chunk_logprobs.top_logprobs:
                                    logprobs_buffer.extend(chunk_logprobs.top_logprobs)
    
                        trigger = (
                            ('\\boxed' in new_text) 
                            or ('assistantfinal' in new_text)
                            or (re.search(r'(?i)\banswer\b', new_text) and re.search(r'\d', new_text))
                        )
                        if trigger:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
    
                            if answer is not None:
                                final_answer = answer
                                break
    
                finally:
                    stream.close()
                
                self._debug_resp(attempt_log, attempt_index, turn_i, full_response_text)

                if final_answer is not None:
                    break
    
                if not token_buffer:
                    break
                
                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    break
    
                if last_message.recipient == 'python':
                    state.python_calls += 1
                    raw_script = last_message.content[0].text
                    final_script = _rewrite_code(raw_script)
                    # execute the code in jupyter sandbox
                    tool_output = local_tool.process_sync_plus(final_script)
                    # should_stop, directives = error_prompter.process(tool_output, final_script)
                    # sanitized_output = error_prompter.format(tool_output, directives)
                    sanitized_output = tool_output

                    # return [self._make_response(output, channel=channel)]
                    tool_responses = [local_tool.make_response(sanitized_output, last_message.channel)]
                    response_text = tool_responses[0].content[0].text
    
                    # now we have 3 "output": tool_output, sanitized_output, response_text...
                    # tool_output -> sanitized_output -> response_text
                    has_error = tool_output.startswith('[ERROR]') or 'Traceback' in tool_output or 'Error:' in tool_output

                    self._debug_python(attempt_log, attempt_index, turn_i, raw_script, final_script, tool_output, sanitized_output, has_error)

                    if has_error: 
                        state.python_errors += 1

                    if tool_output.startswith(ERROR_TIMEOUT): 
                        state.timeout_errors += 1

                    self._update_counter(sanitized_output)
                    conversation.messages.extend(tool_responses)
    
        except Exception as exc:
            state.python_errors += 1
            if self.cfg.debug:
                attempt_log.append(f"\n**EXCEPTION:** {str(exc)}\n")
                print(f"EXCEPTION: {str(exc)}")
    
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)
    
        mean_entropy = compute_mean_entropy(self.cfg.entropy_computer_ver, logprobs_buffer)
        attempt_elapsed = time.time() - attempt_start
        attempt_time = _fmt_time(attempt_elapsed)
        if self.cfg.debug:
            state_json = _format_markdown(json.dumps(state, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2), 'json')
            attempt_log.appendleft(f"Attempt state:\n{state_json}\n")
            attempt_log.appendleft(f"Attempt spent time: **{attempt_time}**\n")
            attempt_log.appendleft(f"## Attempt {attempt_index + 1}\n")

        return {
            'Attempt': attempt_index + 1, 
            'Response Length': state.total_tokens, 
            'Python Calls': state.python_calls, 
            'Python Errors': state.python_errors, 
            'Timeout Errors': state.timeout_errors,
            'Entropy': mean_entropy, 
            'Answer': final_answer,
            'Time': attempt_time,
            'TPS': 1 if attempt_elapsed <= 1e-8 else round(state.total_tokens/attempt_elapsed, 2),
            '_Log': "\n".join(attempt_log),
            '_Elapsed': attempt_elapsed,
            '_PyVerifyCounts': state.py_num_counter,
        }
    
    def _select_answer(self, detailed_results: list, py_verify_global: Counter | None = None) -> int:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)
        py_verify_global = py_verify_global or Counter()

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']
            
            if answer is not None:
                weight = 1.0 / max(entropy, 1e-9)
                bonus = 0.0
                if getattr(self.cfg, "enable_py_verify_vote", False):
                    bonus = float(getattr(self.cfg, "py_verify_weight", 0.0)) * float(py_verify_global.get(int(answer), 0))

                weight += bonus
                answer_weights[answer] += weight
                answer_votes[answer] += 1

        scored_answers = []
        for answer, total_weight in answer_weights.items():
            scored_answers.append({
                'answer': answer, 
                'votes': answer_votes[answer], 
                'score': total_weight
            })

        scored_answers.sort(key=lambda x: x['score'], reverse=True)
        vote_data = []

        for item in scored_answers:
            a = int(item['answer'])
            vote_data.append((
                item['answer'], 
                item['votes'], 
                item['score'],
                int(py_verify_global.get(a, 0))
            ))

        vote_dataframe = pd.DataFrame(
            vote_data, 
            columns=['Answer', 'Votes', 'Score', 'PyCount']
        ).round({'Score': 3})
        display(vote_dataframe)
        
        final_answer = scored_answers[0]['answer'] if scored_answers else 0
        print(f'\nFinal Answer: {final_answer}\n')
        return vote_dataframe, final_answer

    def solve_problem(self, problem: str, problem_id: str = 'UNK') -> int:
        print(f'\nProblem: {problem}\n')
        p_start = time.time()
        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout
    
        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)
    
        deadline = time.time() + budget
    
        print(f'Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n')
    
        tasks = []
    
        for attempt_index in range(self.cfg.attempts):
            tasks.append((self.cfg.system_prompt, attempt_index))
    
        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)
    
        try:
            futures = []
            for (system_prompt, attempt_index) in tasks:
                future = executor.submit(
                    self._process_attempt, 
                    problem, 
                    system_prompt, 
                    attempt_index, 
                    stop_event, 
                    deadline,
                    problem_id,
                )
                futures.append(future)
    
            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)
                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])
    
                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break
    
                except Exception as exc:
                    print(f'Future failed: {exc}')
                    continue
    
        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)
    
        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            results_dataframe['Entropy'] = results_dataframe['Entropy'].round(3)
            results_dataframe['Answer'] = results_dataframe['Answer'].astype('Int64')
            
            cols = [c for c in results_dataframe.columns if not c.startswith('_')]
            display(results_dataframe[cols])

            stat_df = pd.DataFrame(stat_summary(detailed_results))
            display(stat_df)
    
        if not valid_answers:
            print('\nResult: 0\n')
            df_votes, final_answer = pd.DataFrame(columns=['Answer', 'Votes', 'Score']), 0
        else:
            py_verify_global = Counter()
            for r in detailed_results:
                d = r.get('_PyVerifyCounts') or {}
                py_verify_global.update({int(k): int(v) for k, v in d.items()})
            df_votes, final_answer = self._select_answer(detailed_results, py_verify_global)
        
        if self.cfg.debug:
            p_end = time.time()
            p_time = _fmt_time(p_end - p_start)
            self.logger.write_debug_logs(detailed_results, df_votes, problem, problem_id, p_time)
        return final_answer

    def __del__(self):
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass

_delete("server")
server = AIMO3Server(CFG)

_delete("solver")
solver = AIMO3Solver(CFG)

predict_answers = {}

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    global predict_answers
    id_value = id_.item(0)
    question_text = question.item(0)
    gc.disable()
    p_start = time.time()
    final_answer = solver.solve_problem(question_text, problem_id=str(id_value))
    p_end = time.time()
    p_time = p_end - p_start
    predict_answers[id_value] = {'id': id_value, 'answer': final_answer, 'time': p_time, 'time_str': _fmt_time(p_time)}
    gc.enable()
    gc.collect()
    return pl.DataFrame({'id': id_value, 'answer': final_answer})

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

def test(csv: str = 'test.csv'):
    global predict_answers

    def_csv = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
    csv = csv or def_csv

    if not os.path.exists(csv):
        csv = f'/kaggle/input/aimo-p3-hard/{csv}'
    if not os.path.exists(csv):
        csv = def_csv

    CFG.debug = True
    t_start = time.time()
    inference_server.run_local_gateway((csv,))
    t_end = time.time()
    t_time = t_end - t_start

    df = pd.read_csv(csv)
    real_answers = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}
    correct_count = 0
    total_count = df.shape[0]
    # Check accuracy if ground truth available
    for id in predict_answers:
        pa = predict_answers[id]
        if id in real_answers:
            real_answer = real_answers[id]
            is_correct = (pa['answer'] == real_answer)
            if is_correct:
                correct_count += 1
            status = "✅" if is_correct else "❌"
            print(f"Problem {id}: ({pa['time_str']}) -- Predict Answer: {pa['answer']} | Ground Truth: {real_answer} | {status}")
        else:
            print(f"Problem {id}: ({pa['time_str']}) -- Predict Answer: {pa['answer']}")
    
    df2 = pl.DataFrame(list(predict_answers.values()))
    stats = df2.select([
        pl.col("time").max().alias("max_time"),
        pl.col("time").min().alias("min_time"),
        pl.col("time").mean().alias("avg_time"),
    ])

    max_id = df2.filter(pl.col("time") == stats["max_time"][0]).select("id").item()
    min_id = df2.filter(pl.col("time") == stats["min_time"][0]).select("id").item()

    print(f"📊 Total {total_count} problems, ⏱️ total time {_fmt_time(t_time)}; "
          f"Running Accuracy 🎯: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)\n"
          f"Max time: {_fmt_time(stats['max_time'][0])} (id={max_id}); "
          f"Min time: {_fmt_time(stats['min_time'][0])} (id={min_id}); "
          f"Avg time: {_fmt_time(stats['avg_time'][0])}\n")

# test('test.csv')
# test('easy2.csv')
# test('hard3.csv')
# test('peer2.csv')
# test('reference.csv')
# test('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv')
# test('p5.csv')
# test('p9.csv')
# test('p10.csv')

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    CFG.debug = False
    inference_server.serve()

