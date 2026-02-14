# %pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

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
        'openai_harmony'
    ], 
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=True)

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
from typing import Optional
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

    attempts_mode = "serial"
    serial_context_char_limit = 1536
    serial_plan_max_tokens = 384
    serial_aux_temperature = 0.2

    system_prompt = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
        'Use \\boxed{} exactly once at the very end (never for intermediate results).'
        'If you cannot finish within time, output your best verified result anyway as \\boxed{N}.'
    )

    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results. '
        'Python safety: \n'
        '- never compute huge integers directly. \n'
        '- ALWAYS use modular pow(a, e, mod) for big exponents; NEVER call pow(a, huge_e) without mod (or mod=None). Use sympy.factorint(n) when divisors matter. \n'
        '- Never use while True (must have explicit bounds). \n'
        '- Complexity budget: keep each Python call fast (<~2 seconds); start with small bounds and scale up only if needed; \n'
        '- avoid large nested loops or wide scans—if a scan times out, reduce the bound or change the method (use modular/factorization/sieving). \n'
        '- Use explicit namespaces for number theory helpers (math.gcd/math.lcm or sympy.gcd); do not use bare gcd/lcm names. \n'
        '- Dict preview must use list(d.items())[:k] (never d[:k]). \n'
        'Timeout-avoidance rules:\n'
        '- Batch work: do NOT call python repeatedly for small steps; write one cell that computes all needed values.\n'
        '- Before any scan/loop: start with a tiny bound (<=200 or <=2000), time it, then expand by x2/x3 only if fast.\n'
        '- If you see "[ERROR] Execution timed out", do NOT rerun the same code; shrink bounds or add caching.\n'
        '- If computing many ratios/candidates, use caching and early-break; avoid list comprehensions calling expensive funcs.\n'
    )

    preference_prompt = (
        'You have access to `math`, `numpy` and `sympy` to solve the problem.'
        'Prefer verifiable approaches: reduce to modular arithmetic / factorization / small candidate sets. '
        'If an argument depends on choosing the best among many integers, define a candidate set and a coverage strategy (prove a bound or do a bounded scan + verification).'
    )

    # --- NEW: planner (separate session) ---
    planner_system_prompt = (
        'You are an expert IMO problem-solving PLANNER. '
        'Your job is to produce a short plan to guide another solver. '
        'Strict rule: do NOT state any computed values for g(c), p, q, or the remainder. '
        'Do NOT solve the problem. Do NOT output any final answer or \\boxed{}. '
        "Do NOT include any concluding sentence of the form 'therefore the answer is ...' or 'so remainder is ...'. "
        'You may mention problem constants (e.g., 2025, 2025!, M) only as symbols.'
        'Do NOT write Python code. Output must be concise and actionable.'
    )

    planner_prompt = (
        'Output EXACTLY this template, no extra text:\n'
        'PLAN:\n'
        '- ...\n'
        '- ...\n'
        '- ...\n'
        '- ...\n'
        '- ...\n'
        'DIGEST:\n'
        '- ...\n'
        '<<<END>>>\n'
        'Rules:\n'
        '- PLAN has 5-8 bullets, each <=120 chars.\n '
        '- Each PLAN bullet must be an action (verb-first), not a conclusion.\n '
        '- DIGEST is exactly 1 bullet, <=256 chars.\n '
        '- The line "<<<END>>>" must be on its own line (no other text on that line).\n '
        '- "PLAN:" and "DIGEST:" must NOT be prefixed by "-" or "*". They must be standalone headers.\n '
        '- Plan only: do NOT compute the final answer, do NOT include any final numeric result, do NOT write \\boxed{}.\n '
        '- No explanations, no meta talk, no code.n '
        '- Do NOT include the words: analysis, final, assistant, user, rewrite, template.'
    )

    planner_digest_max_chars = 256
    planner_history_keep = 8
    planner_sanitize = True

    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

    # not working??!!
    # served_model_name = 'gpt-oss-sft'
    # model_path = '/kaggle/input/gpt-oss-sft-aimo3/transformers/default/1'

    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17400
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 10    # 6
    sandbox_timeout = 3

    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 2
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02
    enable_error_penalty = True
    error_penalty_lambda = 0.7
    debug = True
    debug_req = True
    debug_resp = True
    debug_limit = 3000
    debug_cols = ['Log', 'Plan', 'PlanRaw', 'PlanSanitized', 'PlanDigest']

set_seed(CFG.seed)


class AIMO3Template:

    def __init__(self):
        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


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

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout

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
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()


class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
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

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]


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

    def format_markdown(self, text: str, mode: str = "quote") -> str:
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

    def log_planner_block(self, plan_raw: str, plan_sanitized: str, plan_digest: str) -> str:
        raw_snip = self.get_debug_snippet(plan_raw)
        san_snip = self.get_debug_snippet(plan_sanitized)
        digest = plan_digest.strip()

        out = []
        out.append("### Planner Output (raw)\n")
        out.append(self.format_markdown(raw_snip, mode="text"))
        out.append("### Planner Output (sanitized)\n")
        out.append(self.format_markdown(san_snip, mode="text"))
        out.append("### Plan Digest\n")
        out.append(self.format_markdown(digest, mode="text"))
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
            final_log_content.append(f"**Problem:**\n{self.format_markdown(problem)}\n")
            final_log_content.append(f"**system_prompt:**\n{self.format_markdown(self.cfg.system_prompt)}\n")
            final_log_content.append(f"**tool_prompt:**\n{self.format_markdown(self.cfg.tool_prompt)}\n")
            final_log_content.append(f"**preference_prompt:**\n{self.format_markdown(self.cfg.preference_prompt)}\n")
            final_log_content.append(f"**planner_system_prompt:**\n{self.format_markdown(self.cfg.planner_system_prompt)}\n")
            final_log_content.append(f"**planner_prompt:**\n{self.format_markdown(self.cfg.planner_prompt)}\n")
            final_log_content.append(f"attempts_mode: **{self.cfg.attempts_mode}**, served_model_name: **{self.cfg.served_model_name}**\n")
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


class AIMO3Planner:
    """
    Robust planner for Harmony-format models:
    - Collect token_ids and parse via harmony encoding to avoid 'assistantfinal' artifacts.
    - Enforce PLAN/DIGEST template with at most one repair.
    - Never return empty digest; never return non-bullet long paragraphs as plan.
    """

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.base_url = f"http://localhost:{port}/v1"
        self.api_key = "sk-local"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

    # ----------------- helpers -----------------

    def _build_history_block(self, history: list[dict]) -> str:
        if not history:
            return "ATTEMPT HISTORY: (none)\n"
        keep = getattr(self.cfg, "planner_history_keep", 8)
        maxc = getattr(self.cfg, "planner_digest_max_chars", 256)
        lines = ["ATTEMPT HISTORY (structured):"]
        for r in history[-keep:]:
            dig = (r.get("PlanDigest") or "").replace("\n", " ").strip()
            if len(dig) > maxc:
                dig = dig[:maxc] + "..."
            lines.append(
                f"- Attempt {r.get('Attempt')}: "
                f"Answer={r.get('Answer')}, Entropy={float(r.get('Entropy', 1e9)):.3f}, "
                f"PyCalls={int(r.get('Python Calls', 0) or 0)}, PyErr={int(r.get('Python Errors', 0) or 0)}; "
                f"PlanDigest={dig}"
            )
        return "\n".join(lines) + "\n"

    def _render_prompt_ids(self, system_prompt: str, user_text: str):
        sys_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.LOW)
            .with_tools(ToolNamespaceConfig(name="none", description="", tools=[]))
        )
        messages = [
            Message.from_role_and_content(Role.SYSTEM, sys_content),
            Message.from_role_and_content(Role.USER, user_text),
        ]
        conv = Conversation.from_messages(messages)
        return self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)

    def _decode_from_token_ids(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        msgs = self.encoding.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
        # concatenate all assistant text contents
        parts = []
        for m in msgs:
            if not m.content:
                continue
            for c in m.content:
                if hasattr(c, "text") and c.text:
                    parts.append(c.text)
        return "".join(parts).strip()

    def _stream_completion(self, system_prompt: str, user_text: str, seed: int, max_tokens: int, temperature: float):
        prompt_ids = self._render_prompt_ids(system_prompt, user_text)
        stream = self.client.completions.create(
            model=self.cfg.served_model_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            prompt=prompt_ids,
            seed=int(seed),
            stream=True,
            extra_body={
                "min_p": self.cfg.min_p,
                "stop_token_ids": self.stop_token_ids,
                "return_token_ids": True,
            },
        )

        token_buf = []
        text_buf = []
        try:
            for chunk in stream:
                # token_ids is the reliable path for harmony parsing
                tids = getattr(chunk.choices[0], "token_ids", None)
                if tids:
                    token_buf.extend(tids)
                t = chunk.choices[0].text
                if t:
                    text_buf.append(t)
        finally:
            stream.close()

        # raw text is only for debugging
        raw_text = "".join(text_buf).strip()
        parsed_text = self._decode_from_token_ids(token_buf)
        return parsed_text, raw_text

    def _bulletize(self, text: str, max_lines: int = 8) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        out = []
        for ln in lines[:max_lines]:
            ln = re.sub(r"^\s*(analysis|final|commentary)\s*", "", ln, flags=re.IGNORECASE).strip()
            if not ln:
                continue
            if not ln.startswith(("-", "*", "•")):
                ln = "- " + ln
            out.append(ln)
        return "\n".join(out).strip()

    def _strip_answerish_lines(self, plan_part: str) -> str:
        lines = plan_part.splitlines()
        bad_kw = ("\\boxed", "remainder", "mod", "final answer", "=", "so each", "therefore", "template", "assistantfinal", "assistant")
        out = []
        for ln in lines:
            low = ln.lower()
            # 只过滤“答案型总结句”，保留正常数学等式的可能性会有误伤，但在 planner 场景利大于弊
            if any(k in low for k in bad_kw) and any(ch.isdigit() for ch in ln):
                continue
            out.append(ln)
        return "\n".join(out).strip()

    def _extract_plan_and_digest(self, text: str) -> tuple[str, str]:
        """
        Accept headers with optional bullet prefixes:
          PLAN: or - PLAN:
          DIGEST: or - DIGEST:
        """
        t = (text or "").strip()
        if not t:
            return "", ""

        # normalize <<<END>>> cut
        t = re.split(r"(?m)^\s*<<<END>>>\s*$", t)[0].strip()

        # find headers (allow optional bullet prefix)
        plan_hdr = re.search(r"(?im)^\s*(?:[-*•]\s*)?PLAN\s*:\s*$", t)
        dig_hdr = re.search(r"(?im)^\s*(?:[-*•]\s*)?DIGEST\s*:\s*$", t)

        if plan_hdr and dig_hdr and dig_hdr.start() > plan_hdr.end():
            plan_block = t[plan_hdr.end():dig_hdr.start()].strip()
            dig_block = t[dig_hdr.end():].strip()
        else:
            # fallback: try split by substring if headers are inline
            if "DIGEST:" in t:
                left, right = t.split("DIGEST:", 1)
                plan_block = left
                dig_block = right
                plan_block = re.sub(r"(?im)^\s*(?:[-*•]\s*)?PLAN\s*:\s*", "", plan_block).strip()
            else:
                plan_block, dig_block = t, ""

        # keep only bullet lines for plan
        plan_lines = []
        for ln in plan_block.splitlines():
            ln = ln.strip()
            if ln.startswith(("-", "*", "•")):
                plan_lines.append(ln)
        plan_part = "\n".join(plan_lines).strip()

        # digest: first bullet line
        digest_line = ""
        for ln in dig_block.splitlines():
            ln = ln.strip()
            if ln.startswith(("-", "*", "•")):
                digest_line = ln.lstrip("-*• ").strip()
                break

        plan_part = re.sub(r"(?i)\b<<<END>>>\b", "", plan_part).strip()
        digest_line = re.sub(r"(?i)\s*\b<<<END>>>\b\s*$", "", digest_line).strip()

        return plan_part, digest_line

    def _make_digest_fallback(self, plan_text: str) -> str:
        maxc = getattr(self.cfg, "planner_digest_max_chars", 256)
        lines = [ln.strip() for ln in (plan_text or "").splitlines() if ln.strip()]
        bullets = [ln.lstrip("-*• ").strip() for ln in lines if ln.startswith(("-", "*", "•"))]
        s = (bullets[0] if bullets else "").strip()
        if not s:
            s = "Try a different approach; enforce small scans + modular checks + caching."
        return s[:maxc].strip()

    def _is_good(self, plan_part: str, digest: str) -> bool:
        if not plan_part:
            return False
        nbul = sum(1 for ln in plan_part.splitlines() if ln.strip().startswith(("-", "*", "•")))
        if nbul < 5:
            return False
        return bool(digest.strip())

    # ----------------- main API -----------------

    def gen_plan(self, problem_text: str, history: list[dict], attempt_index: int):
        # NOTE: planner should be short; do NOT set serial_plan_max_tokens too large.
        max_tokens = min(int(self.cfg.serial_plan_max_tokens), 512)
        temp = float(getattr(self.cfg, "serial_aux_temperature", 0.7))

        user_prompt = (
            f"{self.cfg.planner_prompt}\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"{self._build_history_block(history)}\n"
            f"Output now."
        )

        seed = int((self.cfg.seed + 777) * (attempt_index + 1) ** 2)
        parsed_text, raw_text = self._stream_completion(
            system_prompt=self.cfg.planner_system_prompt,
            user_text=user_prompt,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temp,
        )

        plan_part, digest = self._extract_plan_and_digest(parsed_text)

        # one repair if bad
        if not self._is_good(plan_part, digest):
            repair_user = (
                "FORMATTER TASK. Output ONLY the required template.\n"
                "First line: PLAN:\n"
                "Then 5-8 lines starting with '- '\n"
                "Then: DIGEST:\n"
                "Then exactly one '- ' line (<=256 chars)\n"
                "Then: <<<END>>>\n\n"
                f"PROBLEM:\n{problem_text}\n"
            )
            parsed_text2, raw_text2 = self._stream_completion(
                system_prompt=self.cfg.planner_system_prompt,
                user_text=repair_user,
                seed=seed + 1,
                max_tokens=256,
                temperature=0.0,
            )
            parsed_text = parsed_text2 or parsed_text
            raw_text = raw_text2 or raw_text
            plan_part, digest = self._extract_plan_and_digest(parsed_text)

        # plan_part = self._strip_answerish_lines(plan_part)
        # program-side hard fallback (never return garbage paragraphs)
        if not plan_part.strip():
            plan_part = self._bulletize(parsed_text) or self._bulletize(problem_text)
        if not digest.strip():
            digest = self._make_digest_fallback(plan_part)

        # truncate plan only (digest stays useful)
        # plan_part = plan_part[: self.cfg.serial_context_char_limit].strip()
        # if len(digest) > self.cfg.planner_digest_max_chars:
        #     digest = digest[: self.cfg.planner_digest_max_chars].strip()

        # return both parsed and raw for logging
        plan_sanitized = plan_part  # already bullet-only; treat as sanitized
        plan_raw = raw_text or parsed_text
        return plan_part, digest, plan_raw, plan_sanitized


class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()

        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50
        self.logger = AIMO3Logger(cfg)
        self.planner = AIMO3Planner(cfg)

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
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
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

    def _fmt_time(self, seconds: float) -> str:
        s = int(round(max(0.0, seconds)))
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float,
        problem_id: str,
        plan: str = "",
        plan_digest: str = "",
        plan_raw: str = "",
        plan_sanitized: str = "",
    ) -> dict:
        ## DEBUG
        attempt_log = deque([])
        attempt_start = time.time()

        if stop_event.is_set() or time.time() > deadline:
            print(f"Problem: {problem_id} TIMEOUT!")
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
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout, 
                tool_prompt=self.cfg.tool_prompt, 
                sandbox=sandbox
            )
            encoding = self.encoding

            aug = ""
            if plan:
                aug += f"\n\n=== CURRENT ATTEMPT PLAN ===\n{plan}\n"
            if aug:
                aug += "\nFollow the plan.\n"
            full_problem = problem + aug
            if self.cfg.debug and self.logger:
                attempt_log.append(self.logger.log_planner_block(plan_raw, plan_sanitized, plan_digest))
                attempt_log.append("### Planner Augmentation\n")
                attempt_log.append(f"{self.logger.format_markdown(aug, mode='text')}\n")

            messages = self.template.apply_chat_template(
                system_prompt,
                full_problem, # problem,
                local_tool.tool_config
            )

            conversation = Conversation.from_messages(messages)

            for turn_i in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                ## DEBUG
                if self.cfg.debug and self.cfg.debug_req:
                    # convert prompt_ids (tensors) back to readable text
                    # which includes LLM special symbols like <|im_start|>
                    full_request_text = encoding.decode(prompt_ids)

                    snippet = self.logger.get_debug_snippet(full_request_text)
                    formatted_req = self.logger.format_markdown(snippet)
                    attempt_log.append(f"### Turn {turn_i} - Raw Request to Model:")
                    attempt_log.append(formatted_req)

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

                ## DEBUG
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
                            ## DEBUG
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

                ## DEBUG
                if self.cfg.debug and full_response_text:
                    attempt_log.append(f"### Turn {turn_i} - Model Response:")
                    formatted_resp = self.logger.format_markdown(full_response_text)
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
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text

                    ## DEBUG
                    if self.cfg.debug:
                        code_content = last_message.content[0].text
                        attempt_log.append(f"### Turn {turn_i} - Python Call:")
                        attempt_log.append(f"```python\n{code_content}\n```\n")

                        attempt_log.append(f"### Turn {turn_i} - Python Output:")
                        snippet_out = self.logger.get_debug_snippet(response_text)
                        formatted_out = self.logger.format_markdown(snippet_out, mode="text")
                        attempt_log.append(f"{formatted_out}\n")

                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

        except Exception as exc:
            python_errors += 1
            ## DEBUG
            if self.cfg.debug:
                attempt_log.append(f"\n**EXCEPTION:** {str(exc)}\n")

        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        mean_entropy = self._compute_mean_entropy(logprobs_buffer)
        attempt_elapsed = time.time() - attempt_start
        attempt_time = self._fmt_time(attempt_elapsed)
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
            'Plan': plan,
            'PlanDigest': plan_digest,
            'PlanRaw': plan_raw,
            'PlanSanitized': plan_sanitized,
            'Log': "\n".join(attempt_log),
            'Time': attempt_time
        }

    def _select_answer(self, detailed_results: list) -> int:
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']
            py_err = int(result.get('Python Errors', 0) or 0)

            if answer is not None:
                weight = 1.0 / max(entropy, 1e-9)
                if self.cfg.enable_error_penalty:
                    weight *= 1.0 / (1.0 + self.cfg.error_penalty_lambda * py_err)

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
            return vote_dataframe, 0

        final_answer = scored_answers[0]['answer']    
        print(f'\nFinal Answer: {final_answer}\n')
        return vote_dataframe, final_answer

    def solve_problem(self, problem: str, problem_id: str = "UNK") -> int:
        print(f'\nProblem: {problem}\n')
        problem_start = time.time()

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

        if self.cfg.attempts_mode == "serial":
            # ===== SERIAL MODE =====

            # program-side history ONLY for planner (stable, structured)
            planner_history = []  # list of dicts with Attempt/PlanDigest/Answer/PyCalls/PyErr/Entropy

            for attempt_index in range(self.cfg.attempts):
                if time.time() > deadline:
                    break

                plan, plan_digest, plan_raw, plan_san = self.planner.gen_plan(problem_text=problem, history=planner_history, attempt_index=attempt_index)

                result = self._process_attempt(
                    problem=user_input,
                    system_prompt=self.cfg.system_prompt,
                    attempt_index=attempt_index,
                    stop_event=stop_event,
                    deadline=deadline,
                    problem_id=problem_id,
                    plan=plan,
                    plan_digest=plan_digest,
                    plan_raw=plan_raw,
                    plan_sanitized=plan_san,
                )

                detailed_results.append(result)
                if result.get("Answer") is not None:
                    valid_answers.append(result["Answer"])

                # update planner history (structured)
                planner_history.append({
                    "Attempt": result.get("Attempt", attempt_index + 1),
                    "Answer": result.get("Answer"),
                    "Python Calls": result.get("Python Calls", 0),
                    "Python Errors": result.get("Python Errors", 0),
                    "Entropy": result.get("Entropy", float("inf")),
                    "PlanDigest": plan_digest,
                })

                # early stop (same logic you already use)
                counts = Counter(valid_answers).most_common(1)
                if counts and counts[0][1] >= self.cfg.early_stop:
                    break
                # if attempt_index % 2 == 1: gc.collect()
        else:
            # ===== PARALLEL MODE (retained) =====
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
                        problem_id
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

        problem_elapsed = time.time() - problem_start
        problem_time = self._fmt_time(problem_elapsed)
        if not valid_answers:
            print('\nResult: 0\n')
            vote_data, final_answer = pd.DataFrame(columns=['Answer', 'Votes', 'Score']), 0
        else:
            vote_data, final_answer = self._select_answer(detailed_results)

        self.logger.write_debug_logs(detailed_results, vote_data, problem, problem_id, problem_time)
        return final_answer

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()

        if hasattr(self, 'log_file'):
            self.log_file.close()

        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()

                except Exception:
                    pass


if "solver" in globals(): del solver
solver = AIMO3Solver(CFG)


def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_value = id_.item(0)
    question_text = question.item(0)
    gc.disable()
    final_answer = solver.solve_problem(question_text, problem_id=str(id_value))
    gc.enable()
    gc.collect()
    return pl.DataFrame({'id': id_value, 'answer': final_answer})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    CFG.debug = False
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        # ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
        # ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv',)
        # ('/kaggle/input/aimo-p3-hard/test2.csv',)
        # ('/kaggle/input/aimo-p3-hard/test3.csv',)
        # ('/kaggle/input/aimo-p3-hard/p5.csv',)
        ('/kaggle/input/aimo-p3-hard/p10.csv',)
    )

