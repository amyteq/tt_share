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

    # --- NEW: planner (separate session) ---
    planner_system_prompt = (
        'You are an expert IMO problem-solving PLANNER. '
        'Your job is to produce a short plan to guide another solver. '
        'Do NOT solve the problem. Do NOT output any final answer or \\boxed{}. '
        'Do NOT write Python code. Output must be concise and actionable.'
    )

    planner_prompt = (
        'Output MUST follow this exact template (no extra text):\n'
        'PLAN:\n'
        '- <bullet 1>\n'
        '- <bullet 2>\n'
        '- <bullet 3>\n'
        '- <bullet 4>\n'
        '- <bullet 5>\n'
        'PLAN_DIGEST:\n'
        '- <one bullet, <=256 chars>\n'
        'END_PLAN\n'
        'Rules: bullets ONLY; 5-8 PLAN bullets; each bullet <=120 chars; '
        'no math derivations, no meta talk, no code, no \\boxed{}.'
    )

    planner_digest_max_chars = 256
    planner_history_keep = 8
    planner_sanitize = True

    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    server_timeout = 180
    session_timeout = 960

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


class AIMO3Server:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://localhost:{port}/v1'
        self.api_key = 'sk-local'
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
        # self._initialize_kernels()

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
            # return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
            return None

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]

            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')


if "server" in globals(): server
server = AIMO3Server(CFG)


class AIMO3Planner:

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.base_url = f'http://localhost:{port}/v1'
        self.api_key = 'sk-local'
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.template = AIMO3Template()

    def _sanitize(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip().replace("```", "")
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

        bad = (
            "the user asks", "user asks", "summarize", "output only",
            "valid channels", "system_prompt", "tool_prompt", "preference_prompt",
        )

        out = []
        for ln in lines:
            low = ln.lower()
            if any(b in low for b in bad):
                continue
            if "\\boxed" in ln or "boxed" in low:
                continue
            # keep only bullets / headers we expect
            if low.startswith("plan:") or low.startswith("plan_digest:"):
                out.append(ln)
                continue
            if re.match(r"^(\-|\*|•|\d+[\.\)])\s+", ln):
                out.append(ln)
                continue

        return "\n".join(out).strip()

    def _parse_from(self, raw: str):
        # parse sections
        plan_part = raw
        digest_part = ""
        m = re.search(r"(?im)^\s*PLAN_DIGEST\s*:\s*$", raw)
        if m:
            plan_part = raw[: m.start()].strip()
            digest_part = raw[m.end():].strip()

        # remove PLAN: header if present
        plan_part = re.sub(r"(?im)^\s*PLAN\s*:\s*$", "", plan_part).strip()

        # keep plan short (avoid plan blow-up)
        plan_part = plan_part[: self.cfg.serial_context_char_limit].strip()

        # build digest fallback
        digest_line = ""
        if digest_part:
            # first bullet line
            for ln in digest_part.splitlines():
                ln = ln.strip()
                if ln.startswith(("-", "*", "•")):
                    digest_line = ln.lstrip("-*• ").strip()
                    break
        if not digest_line:
            # fallback: first plan bullet
            for ln in plan_part.splitlines():
                ln = ln.strip()
                if ln.startswith(("-", "*", "•")):
                    digest_line = ln.lstrip("-*• ").strip()
                    break

        if len(digest_line) > self.cfg.planner_digest_max_chars:
            digest_line = digest_line[: self.cfg.planner_digest_max_chars].strip()
        return plan_part, digest_line

    def _validate_plan_text(self, text: str) -> tuple[bool, str]:
        t = (text or "").strip()
        if "PLAN:" not in t or "PLAN_DIGEST:" not in t:
            return (False, "missing headers")
        if "END_PLAN" not in t:
            return (False, "missing END_PLAN")
        # count bullets between PLAN and PLAN_DIGEST
        plan_block = t.split("PLAN:", 1)[1].split("PLAN_DIGEST:", 1)[0]
        bullets = [ln for ln in plan_block.splitlines() if ln.strip().startswith(("-", "*", "•"))]
        if len(bullets) < 5:
            return (False, f"too few bullets: {len(bullets)}")
        digest_block = t.split("PLAN_DIGEST:", 1)[1]
        digest_bullets = [ln for ln in digest_block.splitlines() if ln.strip().startswith(("-", "*", "•"))]
        if not digest_bullets:
            return (False, "empty digest")
        return (True, "ok")

    def _make_digest_fallback(self, plan_text: str) -> str:
        # take first 1-2 bullets and compress
        lines = [ln.strip() for ln in (plan_text or "").splitlines()]
        bullets = [ln.lstrip("-*• ").strip() for ln in lines if ln.startswith(("-", "*", "•"))]
        s = (bullets[0] if bullets else "").strip()
        if not s:
            s = "Try a different approach; enforce small scans + modular checks + caching."
        return s[: self.cfg.planner_digest_max_chars].strip()

    def _build_history_block(self, history: list[dict]) -> str:
        # program-side structured history (stable, no prompt pollution)
        if not history:
            return "ATTEMPT HISTORY: (none)\n"

        lines = ["ATTEMPT HISTORY (structured):"]
        for r in history[-self.cfg.planner_history_keep:]:
            digest = (r.get("PlanDigest") or "").replace("\n", " ").strip()
            if len(digest) > self.cfg.planner_digest_max_chars:
                digest = digest[: self.cfg.planner_digest_max_chars] + "..."
            lines.append(
                f"- Attempt {r.get('Attempt')}: "
                f"Answer={r.get('Answer')}, Entropy={float(r.get('Entropy', 1e9)):.3f}, "
                f"PyCalls={int(r.get('Python Calls', 0) or 0)}, PyErr={int(r.get('Python Errors', 0) or 0)}; "
                f"PlanDigest={digest}"
            )
        return "\n".join(lines) + "\n"

    def _gen_one_shot_text(
        self,
        user_text: str,
        seed: int,
        max_new_tokens: int,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """single-turn generation (planner / summary etc). tools/sandbox disabled."""
        sp = system_prompt or self.cfg.system_prompt
        temp = self.cfg.serial_aux_temperature if temperature is None else float(temperature)
        dummy_tool_cfg = ToolNamespaceConfig(name="python", description="", tools=[])

        messages = self.template.apply_chat_template(sp, user_text, dummy_tool_cfg)
        conversation = Conversation.from_messages(messages)

        prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        # max_tokens = self.cfg.context_tokens - len(prompt_ids) - self.cfg.buffer_tokens
        max_tokens = self.cfg.serial_plan_max_tokens
        if max_tokens <= 0:
            return ""

        max_tokens = min(max_tokens, max_new_tokens)

        stream = self.client.completions.create(
            model=self.cfg.served_model_name,
            temperature=temp,
            logprobs=None,
            max_tokens=max_tokens,
            prompt=prompt_ids,
            seed=seed,
            stream=True,
            extra_body={
                "min_p": self.cfg.min_p,
                "stop_token_ids": self.stop_token_ids,
                "return_token_ids": True,
            },
        )

        chunks = []
        try:
            for chunk in stream:
                txt = chunk.choices[0].text
                if txt:
                    chunks.append(txt)
        finally:
            stream.close()

        return "".join(chunks).strip()

    def gen_plan(self, problem_text: str, history: list[dict], attempt_index: int) -> tuple[str, str, str]:
        prompt = (
            f"{self.cfg.planner_prompt}\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"{self._build_history_block(history)}\n"
            f"Now produce PLAN and PLAN_DIGEST."
        )

        seed = int((self.cfg.seed + 777) * (attempt_index + 1) ** 2)

        raw0 = self._gen_one_shot_text(
            prompt,
            seed=seed,
            max_new_tokens=self.cfg.serial_plan_max_tokens,
            system_prompt=self.cfg.planner_system_prompt,
            temperature=self.cfg.serial_aux_temperature,
        )

        ok, reason = self._validate_plan_text(raw0)
        if not ok:
            # one-shot repair: rewrite ONLY, no new content
            repair_prompt = (
                "REWRITE the following into the EXACT required template. "
                "Do not add any extra commentary.\n\n"
                "BAD_OUTPUT:\n"
                f"{raw0}\n\n"
                "REQUIRED_TEMPLATE:\n"
                f"{self.cfg.planner_prompt}\n"
            )
            raw0 = self._gen_one_shot_text(
                repair_prompt,
                seed=seed + 1,
                max_new_tokens=256,
                system_prompt=self.cfg.planner_system_prompt,
                temperature=0.0,   # repair 用 0 温度更稳
            ).strip()
        raw = raw0
        if self.cfg.planner_sanitize:
            raw = self._sanitize(raw0)

            # fallback: keep raw (trim) rather than empty
            if not raw.strip():
                raw = raw0[: self.cfg.serial_context_char_limit].strip()

        plan_part, digest_line = self._parse_from(raw)
        if not digest_line.strip():
            digest_line = self._make_digest_fallback(plan_part)

        return plan_part, digest_line, raw0, raw


problem = """Let $n \geq 6$ be a positive integer. We call a positive integer $n$-Norwegian if it has three distinct positive divisors whose sum is equal to $n$. Let $f(n)$ denote the smallest $n$-Norwegian positive integer. Let $M=3^{2025!}$ and for a non-negative integer $c$ define 
\begin{equation*}
    g(c)=\frac{1}{2025!}\left\lfloor \frac{2025! f(M+c)}{M}\right\rfloor.
\end{equation*}
We can write 
\begin{equation*}
    g(0)+g(4M)+g(1848374)+g(10162574)+g(265710644)+g(44636594)=\frac{p}{q}
\end{equation*}
where $p$ and $q$ are coprime positive integers. What is the remainder when $p+q$ is divided by $99991$?"""

planner = AIMO3Planner(CFG)
plan, plan_digest, plan_raw, plan_san = planner.gen_plan(problem, [], attempt_index=1)
print(f"plan:\n{plan}\n\nplan_digest:\n{plan_digest}\n\nplan_raw:\n{plan_raw}\n\nplan_sanitized:\n{plan_san}")

