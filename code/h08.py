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
    ], check=True)

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

    system_prompt = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )

    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results.'
    )

    preference_prompt = (
        'You have access to `math`, `numpy` and `sympy` to solve the problem.'
    )

    attempts_mode = "parallel"
    serial_context_char_limit = 1400
    serial_plan_max_tokens = 200
    serial_summary_max_tokens = 220
    serial_aux_temperature = 0.2

    plan_prompt = (
        'You are about to solve a math problem. First produce a concise plan (5-8 bullets).\n'
        'The plan must mention: (i) key idea, (ii) what to brute force / scan, (iii) what to factor/divisibility-check,\n'
        '(iv) what to verify with Python. Do NOT solve the problem yet.'
    )

    summary_prompt = (
        'Summarize previous attempts (plans + outcomes) into a short guidance for the next attempt.\n'
        'Include: tried approaches, candidate answers seen, common failure modes, and what to try next.\n'
        'Keep it brief and actionable.'
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
    temperature = 1.0
    min_p = 0.02
    debug = True
    debug_req = True
    debug_resp = True
    debug_limit = 3000

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

    def _get_debug_snippet(self, text: str) -> str:
        limit = self.cfg.debug_limit
        if len(text) <= limit:
            return text

        head = text[:100]
        tail_len = limit - 100
        tail = text[-tail_len:]
        return f"{head}\n ... \n{tail}"

    def _format_markdown_content(self, text: str, mode: str = "quote") -> str:
        """
        handle text properly when writing it into markdown format debug logs
        e.g. retain LaTex(MathJax), python etc. formats; avoid broken by leading #
        """
        if not text:
            return ""

        # 1. 处理 # 转义：遍历每一行，如果以 # 开头则 prepend \
        lines = text.split('\n')
        escaped_lines = [f"\\{line}" if line.startswith('#') else line for line in lines]
        processed_text = '\n'.join(escaped_lines)

        # 2. 根据模式进行包装
        if mode in ["markdown", "text", "python"]:
            return f"```{mode}\n{processed_text}\n```\n"
        elif mode == "quote":
            # 每行开头增加 "> "
            return '\n'.join([f"> {line}" for line in escaped_lines]) + "\n"
        elif mode == "":
            return processed_text + "\n"
        else:
            return f"```\n{processed_text}\n```\n"

    def _gen_one_shot_text(self, user_text: str, seed: int, max_new_tokens: int) -> str:
        """
        用同一个 vLLM server 做一次“单轮文本生成”（plan/summary 专用），尽量不触发工具。
        返回拼接后的纯文本（可能包含少量前缀符号，调用处再strip/裁剪）。
        """
        # 不需要sandbox；只需要一个工具配置占位（tools=[]）
        dummy_tool_cfg = ToolNamespaceConfig(name="python", description="", tools=[])

        messages = self.template.apply_chat_template(
            self.cfg.system_prompt,
            user_text,
            dummy_tool_cfg,
        )
        conversation = Conversation.from_messages(messages)

        prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        max_tokens = self.cfg.context_tokens - len(prompt_ids) - self.cfg.buffer_tokens
        if max_tokens <= 0:
            return ""

        max_tokens = min(max_tokens, max_new_tokens)

        stream = self.client.completions.create(
            model=self.cfg.served_model_name,
            temperature=self.cfg.serial_aux_temperature,
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

    def _build_plan(self, problem_text: str, serial_summary: str, attempt_index: int) -> str:
        prompt = (
            f"{self.cfg.plan_prompt}\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
        )
        if serial_summary:
            prompt += f"PREVIOUS SUMMARY:\n{serial_summary}\n\n"

        prompt += "Output only the PLAN bullets.\n"

        seed = int((self.cfg.seed + 17) * (attempt_index + 1) ** 2)
        plan = self._gen_one_shot_text(prompt, seed=seed, max_new_tokens=self.cfg.serial_plan_max_tokens)

        # 裁剪避免膨胀
        return plan[: self.cfg.serial_context_char_limit].strip()

    def _build_summary(self, history: list[dict]) -> str:
        """
        history: 每个元素至少含 Attempt/Plan/Answer/Python Calls/Python Errors/Entropy
        """
        if not history:
            return ""

        # 先做一个“结构化原始摘要”（确定性、可控）
        lines = []
        lines.append("ATTEMPT HISTORY (structured):")
        for r in history[-8:]:  # 最多取最近8次，避免过长
            a = r.get("Answer", None)
            plan = (r.get("Plan") or "").replace("\n", " ").strip()
            if len(plan) > 220:
                plan = plan[:220] + "..."
            lines.append(
                f"- Attempt {r.get('Attempt')}: "
                f"Answer={a}, Entropy={float(r.get('Entropy', float('inf'))):.3f}, "
                f"PyCalls={int(r.get('Python Calls', 0) or 0)}, PyErr={int(r.get('Python Errors', 0) or 0)}; "
                f"Plan={plan}"
            )

        raw = "\n".join(lines)
        raw = raw[: self.cfg.serial_context_char_limit * 2]  # 给模型一点空间再压缩

        # 再让模型按 summary_prompt 压缩成“下一次行动指南”
        prompt = (
            f"{self.cfg.summary_prompt}\n\n"
            f"{raw}\n\n"
            "Output a short NEXT-ATTEMPT GUIDANCE (bullets preferred)."
        )
        seed = int((self.cfg.seed + 97) * (len(history) + 1) ** 2)
        summary = self._gen_one_shot_text(prompt, seed=seed, max_new_tokens=self.cfg.serial_summary_max_tokens)

        return summary[: self.cfg.serial_context_char_limit].strip()

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float,
        serial_summary: str = "",
        plan: str = "",
    ) -> dict:
        ## DEBUG
        attempt_log = []
        if self.cfg.debug:
            attempt_log.append(f"## Attempt {attempt_index + 1}\n")

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

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout, 
                tool_prompt=self.cfg.tool_prompt, 
                sandbox=sandbox
            )

            encoding = self.encoding
            # messages = self.template.apply_chat_template(
            #     system_prompt, 
            #     problem, 
            #     local_tool.tool_config
            # )

            aug = ""
            if serial_summary:
                aug += f"\n\n=== PREVIOUS ATTEMPTS SUMMARY ===\n{serial_summary}\n"
            if plan:
                aug += f"\n\n=== CURRENT ATTEMPT PLAN ===\n{plan}\n"
            if aug:
                aug += "\nFollow the plan. Avoid repeating old failed approaches unless you strengthen verification/coverage.\n"

            full_problem = problem + aug

            messages = self.template.apply_chat_template(
                system_prompt,
                full_problem,
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

                        snippet = self._get_debug_snippet(full_request_text)
                        formatted_req = self._format_markdown_content(snippet)
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
                    formatted_resp = self._format_markdown_content(full_response_text)
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
                        snippet_out = self._get_debug_snippet(response_text)
                        formatted_out = self._format_markdown_content(snippet_out, mode="text")
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

        return {
            'Attempt': attempt_index + 1, 
            'Response Length': total_tokens, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Entropy': mean_entropy, 
            'Answer': final_answer,
            'Plan': plan,
            'Log': "\n".join(attempt_log)
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
            return vote_dataframe, 0

        final_answer = scored_answers[0]['answer']    
        print(f'\nFinal Answer: {final_answer}\n')
        return vote_dataframe, final_answer

    ## DEUBG
    def write_debug_logs(self, detailed_results: list, vote_dataframe: pd.DataFrame, problem: str, problem_id: str = "UNK"):
        if not self.cfg.debug: return

        try:
            summary_lines = ["\n## Summary Stats\n"]
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                cols = [c for c in df.columns if not c in ['Log', 'Plan']]
                summary_lines.append(df[cols].to_markdown(index=False))
                summary_lines.append("\n\n")

            if not vote_dataframe.empty:
                summary_lines.append("## Vote Counts\n")
                summary_lines.append(vote_dataframe.to_markdown(index=False))
                summary_lines.append("\n")

            # 拼接所有内容：Header -> Summary -> 各个 Attempt 的详细日志
            final_log_content = [f"# Problem ID: {problem_id}\n"]
            final_log_content.append(f"**Problem:**\n{self._format_markdown_content(problem)}\n")
            final_log_content.append(f"**system_prompt:**\n{self._format_markdown_content(self.cfg.system_prompt)}\n\n")
            final_log_content.append(f"**tool_prompt:**\n{self._format_markdown_content(self.cfg.tool_prompt)}\n\n")
            final_log_content.append(f"**preference_prompt:**\n{self._format_markdown_content(self.cfg.preference_prompt)}\n\n")

            final_log_content.extend(summary_lines)
            final_log_content.append("\n---\n")

            # 按 Attempt 顺序排序日志
            # 注意：如果 detailed_results 里没有 Log 字段 (比如被删了)，这里要防守一下
            # 但我们在 _process_attempt 里是保证返回 Log 的
            sorted_results = sorted(detailed_results, key=lambda x: x['Attempt'])
            for res in sorted_results:
                log_content = res.get('Log', '')
                if log_content:
                    final_log_content.append(log_content)
                    final_log_content.append("\n---\n")

            # 写入文件
            output_path = f"{problem_id}.md" 
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(final_log_content))
            print(f"Debug log written to {output_path}")

        except Exception as e:
            print(f"Failed to write debug log: {e}")

    def solve_problem(self, problem: str, problem_id: str = "UNK") -> int:
        print(f'\nProblem: {problem}\n')

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

        if self.cfg.attempts_mode.lower() in ["serial"]:
            # ===== SERIAL MODE =====
            serial_summary = ""
            history = []

            for attempt_index in range(self.cfg.attempts):
                if stop_event.is_set() or time.time() > deadline:
                    break

                plan = self._build_plan(problem_text=user_input, serial_summary=serial_summary, attempt_index=attempt_index)

                result = self._process_attempt(
                    user_input,
                    self.cfg.system_prompt,
                    attempt_index,
                    stop_event,
                    deadline,
                    serial_summary=serial_summary,
                    plan=plan,
                )

                detailed_results.append(result)
                history.append(result)

                if result.get("Answer") is not None:
                    valid_answers.append(result["Answer"])

                # early stop（沿用你原逻辑）
                counts = Counter(valid_answers).most_common(1)
                if counts and counts[0][1] >= self.cfg.early_stop:
                    stop_event.set()
                    break

                # 生成下一轮要喂的summary（把 plan+结果压缩）
                serial_summary = self._build_summary(history)

                # 轻量清理，避免串行累积（可选）
                if attempt_index % 2 == 1:
                    gc.collect()
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
                        deadline
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

            cols = [c for c in results_dataframe.columns if not c in ['Log', 'Plan']]
            display(results_dataframe[cols])

        if not valid_answers:
            print('\nResult: 0\n')
            vote_data, final_answer = pd.DataFrame(columns=['Answer', 'Votes', 'Score']), 0
        else:
            vote_data, final_answer = self._select_answer(detailed_results)

        self.write_debug_logs(detailed_results, vote_data, problem, problem_id=problem_id)
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
        # ('/kaggle/input/aimo-p3-hard/test2.csv',)
        ('/kaggle/input/aimo-p3-hard/p10.csv',)
    )

