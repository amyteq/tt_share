#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('pip', "uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'")


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

import gc
import re
import math
import time
import queue
import threading
import contextlib
from collections import deque
from typing import Iterable, Optional
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

class CFG:

    # prompts sometimes seems useless, such as to restrict d[:n], keep it simple?
    # use AST instead to validate/tweak generated codes?
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

    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

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
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    # TODO: why this combination? not temperature <= 0.1 + min_p = 0.95? to EXP
    temperature = 1.0
    min_p = 0.02

    # AST validate & fix
    ast_fix_slice = True
    ast_add_cache = False
    ast_fix_print = True

    # debug
    debug = True
    debug_req = True
    debug_req_full = 20
    debug_resp = True
    debug_limit = 3000
    debug_cols = ['Log', 'Plan', 'PlanRaw', 'PlanSanitized', 'PlanDigest']

set_seed(CFG.seed)


import ast

def _ann_contains_callable(ann: ast.AST) -> bool:
    # covers: Callable, typing.Callable, collections.abc.Callable, etc.
    if ann is None:
        return False
    # Name("Callable")
    if isinstance(ann, ast.Name) and ann.id.lower() == "callable":
        return True
    # Attribute(..., attr="Callable")
    if isinstance(ann, ast.Attribute) and ann.attr.lower() == "callable":
        return True
    # Subscript(Callable[[...], ...])
    if isinstance(ann, ast.Subscript):
        return _ann_contains_callable(ann.value) or _ann_contains_callable(ann.slice)
    # Union / | : just scan children
    for child in ast.iter_child_nodes(ann):
        if _ann_contains_callable(child):
            return True
    return False

def _has_uncacheable_params(fn: ast.FunctionDef) -> bool:
    bad_name_tokens = ("func", "callable", "callback", "fn", "lambda")
    args = fn.args

    # varargs/kwargs => unstable signature; skip
    if args.vararg is not None or args.kwarg is not None:
        return True

    # positional + kwonly args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)

    for a in all_args:
        # name heuristic
        if any(tok in a.arg.lower() for tok in bad_name_tokens):
            return True
        # annotation heuristic
        if _ann_contains_callable(a.annotation):
            return True

    return False

class _Rewriter(ast.NodeTransformer):
    def __init__(self, cfg):
        self.cfg = cfg

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # recurse first
        self.generic_visit(node)

        if not self.cfg.ast_add_cache or _has_uncacheable_params(node):
            return node

        # if already has @memory.cache, do nothing
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Name)
                and dec.value.id == "memory"
                and dec.attr == "cache"):
                return node

        # prepend @memory.cache
        node.decorator_list.insert(
            0,
            ast.Attribute(value=ast.Name(id="memory", ctx=ast.Load()), attr="cache", ctx=ast.Load())
        )
        return node

    def visit_Subscript(self, node: ast.Subscript):
        # rewrite inside first
        self.generic_visit(node)

        if not self.cfg.ast_fix_slice:
            return node

        s = node.slice
        if isinstance(s, ast.Slice) and s.upper is not None:
            # x[:n] / x[a:n] => head(x, n)
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="head", ctx=ast.Load()),
                    args=[node.value, s.upper],
                    keywords=[]
                ),
                node,
            )
        return node

def _to_load(t: ast.AST) -> ast.AST:
    # minimal conversion for assignment targets
    if isinstance(t, ast.Name):
        return ast.Name(id=t.id, ctx=ast.Load())
    if isinstance(t, ast.Tuple):
        return ast.Tuple(elts=[_to_load(e) for e in t.elts], ctx=ast.Load())
    if isinstance(t, ast.List):
        return ast.List(elts=[_to_load(e) for e in t.elts], ctx=ast.Load())
    if isinstance(t, ast.Attribute):
        return ast.Attribute(value=_to_load(t.value), attr=t.attr, ctx=ast.Load())
    if isinstance(t, ast.Subscript):
        return ast.Subscript(value=_to_load(t.value), slice=t.slice, ctx=ast.Load())
    return t  # fallback

def _rewrite_code(code: str, cfg = CFG) -> str:
    tree = ast.parse(code)
    if not tree.body:
        return code

    # 1) rewrite function decorators + slices
    tree = _Rewriter(cfg).visit(tree)
    ast.fix_missing_locations(tree)

    if not cfg.ast_fix_print:
        return ast.unparse(tree)

    # 2) auto print last line if it is assignment
    last = tree.body[-1]
    # a = 1 => print(a)
    if isinstance(last, ast.Assign):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[_to_load(t) for t in last.targets],
                keywords=[]
            )
        )
    # 2+3 => print(2+3)
    if isinstance(last, ast.Expr):
        # Avoid double-print if it's already print(...)
        v = last.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print"):
            tree.body.append(ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[v],
                    keywords=[]
                )
            ))

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

### util methods
def _fmt_time(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def _format_markdown(text: str, mode: str = "quote") -> str:
    if not text:
        return ""
    lines = text.split('\n')
    escaped_lines = [f"\\{line}" if line.startswith('#') else line for line in lines]
    processed_text = '\n'.join(escaped_lines)
    if mode in ["markdown", "text", "python"]:
        return f"```{mode}\n{processed_text}\n```\n"
    if mode == "quote":
        return '\n'.join([f"> {line}" for line in escaped_lines]) + "\n"
    if mode == "":
        return processed_text + "\n"
    return f"```\n{processed_text}\n```\n"

def _delete(name: str):
    if name is not None and name != "" and name in globals(): 
        del globals()[name]


class AIMO3Logger:
    def __init__(self, cfg):
        self.cfg = cfg

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

    def write_debug_logs(self, detailed_results, vote_dataframe, problem, problem_id="UNK", problem_time=""):
        if not self.cfg.debug:
            return
        try:
            summary_lines = ["\n## Summary Stats\n"]
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                cols = [c for c in df.columns if c not in self.cfg.debug_cols]
                summary_lines.append(df[cols].to_markdown(index=False))
                summary_lines.append("\n\n")

            if not vote_dataframe.empty:
                summary_lines.append("## Vote Counts\n")
                summary_lines.append(vote_dataframe.to_markdown(index=False))
                summary_lines.append("\n")

            final_log_content = [f"# Problem ID: {problem_id}\n"]
            final_log_content.append(f"Problem spent time: **{problem_time}**\n\n")
            final_log_content.append(f"**Problem:**\n{_format_markdown(problem)}\n")
            final_log_content.append(f"**system_prompt:**\n{_format_markdown(self.cfg.system_prompt)}\n")
            final_log_content.append(f"**tool_prompt:**\n{_format_markdown(self.cfg.tool_prompt)}\n")
            final_log_content.append(f"**preference_prompt:**\n{_format_markdown(self.cfg.preference_prompt)}\n")
            final_log_content.append(f"**CFG** > temperature: **{self.cfg.temperature}**, "
                                     f"min_p: **{self.cfg.min_p}**, "
                                     f"served_model_name: **{self.cfg.served_model_name}**\n")
            final_log_content.extend(summary_lines)
            final_log_content.append("\n===\n")

            sorted_results = sorted(detailed_results, key=lambda x: x['Attempt'])
            for res in sorted_results:
                log_content = res.get('Log', '')
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

        self._helper_imports = """
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
    \"\"\"
    Solve x**n ≡ a (mod m).
    Returns:
      - smallest solution if all_roots=False
      - sorted list of all solutions modulo m if all_roots=True

    Notes:
      - Uses sympy.ntheory.residue_ntheory.nthroot_mod for prime/prime-power moduli.
      - For composite m, solves each prime-power modulus then combines via CRT.
    \"\"\"
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
"""

        self._helper_methods = """
def head(obj, n=5):
    if isinstance(obj, dict):
        return dict(list(obj.items())[:n])
    try:
        return obj[:n]
    except:
        return obj
"""

        self.execute([self._helper_imports, self._helper_methods])

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

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
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

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

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

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
        self.execute(['%reset -f\n', self._helper_imports, self._helper_methods])

    def __del__(self):
        self.close()


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

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split('\n')
        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code
        if not last_line:
            return code
        if last_line.startswith('#'):
            return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

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

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, code: str, channel: str) -> list[Message]:
        self._ensure_session()
        # final_script = self._ensure_last_print(code)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(code)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=channel)]


class AIMO3Server:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://localhost:{port}/v1'
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


class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://localhost:{port}/v1'
        self.api_key = 'sk-local'
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50
        self.logger = AIMO3Logger(cfg)
        self._initialize_kernels()

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
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)

        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value

            except ValueError:
                pass

        pattern = r'final\s+answer\s+is\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value

            except ValueError:
                pass

        return None

    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
        if not logprobs_buffer:
            return float('inf')

        total_entropy = 0.0
        token_count = 0

        for top_logprobs_dict in logprobs_buffer:            
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

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float,
        problem_id: str,
        plan: str = "",
    ) -> dict:
        attempt_log = deque([])
        attempt_start = time.time()

        if stop_event.is_set() or time.time() > deadline:
            return {
                'Attempt': attempt_index + 1, 
                'Answer': None, 
                'Python Calls': 0, 
                'Python Errors': 0, 
                'Response Length': 0, 
                'Entropy': float('inf'),
                'Log': "\n".join(attempt_log)
            }

        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None

        logprobs_buffer = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)    
            local_tool = AIMO3Tool(self.cfg, sandbox=sandbox)

            encoding = self.encoding
            system_content = (
                SystemContent.new()
                .with_model_identity(system_prompt)
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(local_tool.tool_config)
            )
            messages = [
                Message.from_role_and_content(Role.SYSTEM, system_content),
                Message.from_role_and_content(Role.USER, problem), # problem
            ]
            conversation = Conversation.from_messages(messages)

            for turn_i in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                if self.cfg.debug and self.cfg.debug_req:
                    full_request_text = encoding.decode(prompt_ids)
                    snippet = self.logger.get_debug_snippet(full_request_text)
                    # for Turn 0, 20,... log full request
                    formatted_req = _format_markdown(full_request_text if turn_i % self.cfg.debug_req_full == 0 else snippet)
                    attempt_log.append(f"### Turn {turn_i} - Raw Request to Model:")
                    attempt_log.append(formatted_req)

                # didn't find any timeout, ignore for now
                # req_timeout = max(1.0, deadline - time.time())
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, 
                    temperature=self.cfg.temperature, 
                    logprobs=self.cfg.top_logprobs, 
                    max_tokens=max_tokens, 
                    prompt=prompt_ids, 
                    seed=attempt_seed, 
                    stream=True, 
                    extra_body={
                        'min_p': self.cfg.min_p, 
                        'stop_token_ids': self.stop_token_ids, 
                        'return_token_ids': True
                    }
                )

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
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            if self.cfg.debug and self.cfg.debug_resp:
                                full_response_text += new_text

                            chunk_logprobs = chunk.choices[0].logprobs
                            if chunk_logprobs is not None:
                                if chunk_logprobs.top_logprobs:
                                    logprobs_buffer.extend(chunk_logprobs.top_logprobs)

                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)

                            if answer is not None:
                                final_answer = answer
                                break

                finally:
                    stream.close()

                if self.cfg.debug and full_response_text:
                    attempt_log.append(f"### Turn {turn_i} - Model Response:")
                    formatted_resp = _format_markdown(full_response_text)
                    attempt_log.append(formatted_resp)

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
                    python_calls += 1
                    raw_script = last_message.content[0].text
                    final_script = _rewrite_code(raw_script)
                    tool_responses = local_tool.process_sync_plus(final_script, last_message.channel)
                    response_text = tool_responses[0].content[0].text

                    has_error = response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text
                    if self.cfg.debug:
                        emoji_error = '❌' if has_error else ''
                        attempt_log.append(f"### Turn {turn_i} - Python Raw:")
                        attempt_log.append(_format_markdown(raw_script, mode='python'))
                        attempt_log.append(f"### Turn {turn_i} - Python Call:")
                        attempt_log.append(_format_markdown(final_script, mode='python'))
                        attempt_log.append(f"### Turn {turn_i} {emoji_error} - Python Output:")
                        snippet_out = self.logger.get_debug_snippet(response_text)
                        formatted_out = _format_markdown(snippet_out, mode='text')
                        attempt_log.append(f"{formatted_out}\n")

                    if has_error:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

        except Exception as exc:
            python_errors += 1
            if self.cfg.debug:
                attempt_log.append(f"\n**EXCEPTION:** {str(exc)}\n")
            print(f"EXCEPTION: {str(exc)}")

        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)
        attempt_elapsed = time.time() - attempt_start
        attempt_time = _fmt_time(attempt_elapsed)
        if self.cfg.debug:
            attempt_log.appendleft(f"Attempt spent time: **{attempt_time}**\n")
            attempt_log.appendleft(f"## Attempt {attempt_index + 1}\n")

        return {
            'Attempt': attempt_index + 1, 
            'Response Length': total_tokens, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Entropy': mean_entropy, 
            'Answer': final_answer,
            'Log': "\n".join(attempt_log),
            'Time': attempt_time
        }

    def _select_answer(self, detailed_results: list) -> int:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']

            if answer is not None:
                weight = 1.0 / max(entropy, 1e-9)
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
            vote_data.append((
                item['answer'], 
                item['votes'], 
                item['score']
            ))

        vote_dataframe = pd.DataFrame(
            vote_data, 
            columns=['Answer', 'Votes', 'Score']
        )

        vote_dataframe = vote_dataframe.round({'Score': 3})
        display(vote_dataframe)

        if not scored_answers:
            print('\nFinal Answer: 0\n')
            return 0

        final_answer = scored_answers[0]['answer']    
        print(f'\nFinal Answer: {final_answer}\n')

        return vote_dataframe, final_answer

    def solve_problem(self, problem: str, problem_id: str = 'UNK') -> int:
        print(f'\nProblem: {problem}\n')
        p_start = time.time()
        user_input = f'{problem} {self.cfg.preference_prompt}'

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
                    user_input, 
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

            cols = [c for c in results_dataframe.columns if not c in self.cfg.debug_cols]
            display(results_dataframe[cols])

        if not valid_answers:
            print('\nResult: 0\n')
            vote_data, final_answer = pd.DataFrame(columns=['Answer', 'Votes', 'Score']), 0
        else:
            vote_data, final_answer = self._select_answer(detailed_results)

        p_end = time.time()
        self.logger.write_debug_logs(detailed_results, vote_data, problem, problem_id, _fmt_time(p_end - p_start))
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

def test():
    global predict_answers

    # test_csv = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
    # test_csv = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv'
    # test_csv = '/kaggle/input/aimo-p3-hard/easy2.csv'
    # test_csv = '/kaggle/input/aimo-p3-hard/test2.csv'
    # test_csv = '/kaggle/input/aimo-p3-hard/test3.csv'
    # test_csv = '/kaggle/input/aimo-p3-hard/p5.csv'
    test_csv = '/kaggle/input/aimo-p3-hard/p10.csv'

    t_start = time.time()
    inference_server.run_local_gateway((test_csv,))
    t_end = time.time()
    t_time = t_end - t_start

    df = pd.read_csv(test_csv)
    real_answers = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}
    correct_count = 0
    total_count = 0
    # Check accuracy if ground truth available
    for id in predict_answers:
        pa = predict_answers[id]
        if id in real_answers:
            total_count += 1
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


CFG.ast_add_cache = True

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    CFG.debug = False
    inference_server.serve()
else:
    test()

