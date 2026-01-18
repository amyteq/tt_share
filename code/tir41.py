#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import os

start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 55) * 60  # 4h 55m

TOTAL_TIME = 4 * 60 * 60 + 55 * 60  # 4h 55m
NUM_QUESTIONS = 50
BUFFER_TIME = 60


import subprocess

uninstall_proc = subprocess.Popen(
    ["pip", "uninstall", "--yes", "tensorflow", "matplotlib", "keras", "scikit-learn"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)


get_ipython().run_cell_magic('time', '', '!find /kaggle/usr/lib -type f -print0 | xargs -0 -P 32 -n 500 cat > /dev/null\n')


def cache_model(path, exts=(".bin", ".pt", ".safetensors"), num_workers=None, chunk_mb=256):
    """Pre-read model weight files into OS page cache."""
    import os
    import multiprocessing
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath):
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        raise ValueError(f"No model files found under: {path}")

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    print(f"[cache_model] total read ≈ {gb:.2f} GB in {elapsed:.2f}s")
    return total_bytes


cache_model("/kaggle/input/gpt-oss-120b/transformers/default/1", num_workers=16, chunk_mb=1024)


get_ipython().run_cell_magic('time', '', '# Copy vLLM compile cache if available\nimport os\nif os.path.exists("/kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache"):\n    !mkdir -p /root/.cache/vllm/\n    !cp -r /kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache /root/.cache/vllm/\n')


uninstall_proc.wait()


subprocess.run(["ls", "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"])


os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"


get_ipython().run_cell_magic('writefile', 'local_python_tool.py', '"""Python tool using Jupyter kernel for stateful execution."""\nimport os\nimport queue\nimport threading\nfrom abc import ABC, abstractmethod\nfrom typing import AsyncIterator, Any\nfrom uuid import UUID, uuid4\n\nfrom openai_harmony import (\n    Author,\n    Content,\n    Message,\n    Role,\n    TextContent,\n    ToolNamespaceConfig,\n)\n\n\ndef add_libs(code: str) -> str:\n    """Add common math libraries to code."""\n    return "import math\\nimport numpy as np\\nimport sympy as sp\\nfrom sympy import *\\n" + code\n\n\ndef ensure_last_print(code: str) -> str:\n    """Ensure the last expression is printed."""\n    lines = code.strip().split("\\n")\n    if lines and "print(" not in lines[-1] and "import" not in lines[-1]:\n        if "#" in lines[-1]:\n            lines[-1] = lines[-1].split("#")[0]\n        lines[-1] = "print(" + lines[-1] + ")"\n    return "\\n".join(lines)\n\n\nclass LocalJupyterSession:\n    """Stateful Jupyter kernel session for code execution."""\n\n    # Class-level lock and port counter to avoid port conflicts\n    _port_lock = threading.Lock()\n    _next_port = 50000\n\n    @classmethod\n    def _get_next_ports(cls, count: int = 5) -> list[int]:\n        """Get next available ports for kernel connection."""\n        with cls._port_lock:\n            ports = list(range(cls._next_port, cls._next_port + count))\n            cls._next_port += count\n            return ports\n\n    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0):\n        try:\n            from jupyter_client import BlockingKernelClient, KernelManager\n        except ImportError as exc:\n            raise RuntimeError("jupyter_client package required") from exc\n\n        self._default_timeout = timeout\n        self._owns_kernel = False\n        self._client: BlockingKernelClient\n        self._km: KernelManager | None = None\n\n        if connection_file:\n            from pathlib import Path\n            connection_path = Path(connection_file).expanduser()\n            if not connection_path.exists():\n                raise FileNotFoundError(f"Connection file not found: {connection_path}")\n            client = BlockingKernelClient()\n            client.load_connection_file(str(connection_path))\n            client.start_channels()\n            client.wait_for_ready(timeout=self._default_timeout)\n            self._client = client\n        else:\n            # Allocate unique ports to avoid conflicts when running multiple kernels\n            ports = self._get_next_ports(5)\n            km = KernelManager()\n            km.shell_port = ports[0]\n            km.iopub_port = ports[1]\n            km.stdin_port = ports[2]\n            km.hb_port = ports[3]\n            km.control_port = ports[4]\n            km.start_kernel()\n            client = km.blocking_client()\n            client.start_channels()\n            client.wait_for_ready(timeout=self._default_timeout)\n            self._client = client\n            self._km = km\n            self._owns_kernel = True\n\n    def execute(self, code: str, *, timeout: float | None = None) -> str:\n        """Execute code and return combined stdout/stderr."""\n        client = self._client\n        effective_timeout = timeout or self._default_timeout\n        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)\n\n        stdout_parts: list[str] = []\n        stderr_parts: list[str] = []\n\n        while True:\n            try:\n                msg = client.get_iopub_msg(timeout=effective_timeout)\n            except queue.Empty as exc:\n                raise TimeoutError("Timed out waiting for kernel output.") from exc\n\n            if msg.get("parent_header", {}).get("msg_id") != msg_id:\n                continue\n\n            msg_type = msg.get("msg_type")\n            content = msg.get("content", {})\n\n            if msg_type == "stream":\n                text = content.get("text", "")\n                if content.get("name") == "stdout":\n                    stdout_parts.append(text)\n                else:\n                    stderr_parts.append(text)\n            elif msg_type == "error":\n                traceback_data = content.get("traceback")\n                if traceback_data:\n                    stderr_parts.append("\\n".join(traceback_data))\n                else:\n                    ename = content.get("ename", "")\n                    evalue = content.get("evalue", "")\n                    stderr_parts.append(f"{ename}: {evalue}".strip())\n            elif msg_type in {"execute_result", "display_data"}:\n                data = content.get("data", {})\n                text = data.get("text/plain")\n                if text:\n                    stdout_parts.append(text if text.endswith("\\n") else f"{text}\\n")\n            elif msg_type == "status" and content.get("execution_state") == "idle":\n                break\n\n        # Drain shell channel\n        while True:\n            try:\n                reply = client.get_shell_msg(timeout=effective_timeout)\n            except queue.Empty as exc:\n                raise TimeoutError("Timed out waiting for execution reply.") from exc\n\n            if reply.get("parent_header", {}).get("msg_id") != msg_id:\n                continue\n\n            reply_content = reply.get("content", {})\n            if reply_content.get("status") == "error":\n                traceback_data = reply_content.get("traceback")\n                if traceback_data:\n                    stderr_parts.append("\\n".join(traceback_data))\n                else:\n                    ename = reply_content.get("ename", "")\n                    evalue = reply_content.get("evalue", "")\n                    stderr_parts.append(f"{ename}: {evalue}".strip())\n            break\n\n        stdout = "".join(stdout_parts)\n        stderr = "".join(stderr_parts)\n\n        if stderr:\n            stdout = f"{stdout.rstrip()}\\n{stderr}" if stdout else stderr\n\n        if not stdout.strip():\n            stdout = "[WARN] No output. Use print() to see results."\n\n        return stdout\n\n    def close(self):\n        import contextlib\n        with contextlib.suppress(Exception):\n            self._client.stop_channels()\n        if self._owns_kernel and self._km is not None:\n            with contextlib.suppress(Exception):\n                self._km.shutdown_kernel(now=True)\n\n    def __del__(self):\n        self.close()\n\n\nclass PythonTool:\n    """Python execution tool using Jupyter kernel."""\n\n    def __init__(self, execution_backend: str | None = None, local_jupyter_timeout: float = 60.0):\n        self._local_jupyter_timeout = local_jupyter_timeout\n        self._execution_lock = threading.Lock()\n        self._jupyter_session: LocalJupyterSession | None = None\n        # Lazy initialization to avoid port conflicts during object creation\n        self._init_lock = threading.Lock()\n\n    def _ensure_session(self):\n        """Lazily initialize the Jupyter session."""\n        if self._jupyter_session is None:\n            with self._init_lock:\n                if self._jupyter_session is None:\n                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)\n\n    @classmethod\n    def get_tool_name(cls) -> str:\n        return "python"\n\n    @property\n    def name(self) -> str:\n        return self.get_tool_name()\n\n    @property\n    def instruction(self) -> str:\n        return """Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."""\n\n    @property\n    def tool_config(self) -> ToolNamespaceConfig:\n        return ToolNamespaceConfig(\n            name=self.get_tool_name(),\n            description=self.instruction,\n            tools=[]\n        )\n\n    def _make_response(self, output: str, channel: str | None = None) -> Message:\n        content = TextContent(text=output)\n        author = Author(role=Role.TOOL, name=self.get_tool_name())\n        message = Message(author=author, content=[content]).with_recipient("assistant")\n        if channel:\n            message = message.with_channel(channel)\n        return message\n\n    def process_sync_plus(self, message: Message) -> list[Message]:\n        """Execute code from message using Jupyter kernel."""\n        self._ensure_session()\n        script = message.content[0].text\n        with self._execution_lock:\n            try:\n                output = self._jupyter_session.execute(script)\n            except TimeoutError as exc:\n                output = f"[ERROR] {exc}"\n        return [self._make_response(output, channel=message.channel)]\n\n    def close(self):\n        if self._jupyter_session is not None:\n            self._jupyter_session.close()\n            self._jupyter_session = None\n\n    def __del__(self):\n        self.close()\n')


import warnings
warnings.simplefilter('ignore')

import re
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd
import polars as pl
from openai import OpenAI
from transformers import set_seed, AutoTokenizer
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
    RenderConversationConfig,
)

from local_python_tool import PythonTool

# Load Harmony encoding for GPT-OSS
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Constants
SEED = 42
set_seed(SEED)
MAX_LEN = 64 * 1024
USE_BUDGET = False
K = 8  # Number of parallel samples

# Inference parameters (same as way-to-30 reference)
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02


class DynamicTimeBudget:
    """Manages dynamic time allocation with rollover from early stopping."""

    def __init__(self, total_time_seconds: float, num_questions: int, buffer_seconds: float = 60):
        self.total_time = total_time_seconds
        self.num_questions = num_questions
        self.buffer = buffer_seconds
        self.start_time = time.time()

        # Available time excluding buffer
        self.available_time = total_time_seconds - buffer_seconds

        # Track time usage
        self.time_used = 0
        self.questions_completed = 0
        self.time_saved = 0  # Accumulated time from early stops

    def get_deadline_for_question(self) -> float:
        """Calculate deadline for current question with rollover time."""
        questions_remaining = self.num_questions - self.questions_completed

        if questions_remaining <= 0:
            return time.time() + 60  # Emergency fallback

        # Base time per remaining question
        time_remaining = self.available_time - self.time_used
        base_time = time_remaining / questions_remaining

        # Add any saved time from early stopping
        allocated_time = base_time + self.time_saved

        # Reset saved time (it's now allocated to this question)
        self.time_saved = 0

        deadline = time.time() + allocated_time

        print(f"⏱️  Allocated {allocated_time:.1f}s for question {self.questions_completed + 1}")
        print(f"   (Base: {base_time:.1f}s, Rollover: {self.time_saved:.1f}s, Remaining: {questions_remaining} questions)")

        return deadline

    def record_question_completion(self, time_spent: float, early_stopped: bool = False):
        """Record completion and calculate time savings."""
        self.time_used += time_spent
        self.questions_completed += 1

        # If early stopped, calculate how much time was saved
        if early_stopped:
            questions_remaining = self.num_questions - self.questions_completed
            if questions_remaining > 0:
                expected_time = (self.available_time - self.time_used + time_spent) / (questions_remaining + 1)
                time_saved = max(0, expected_time - time_spent)
                self.time_saved += time_saved
                print(f"💰 Early stop saved {time_saved:.1f}s (total saved: {self.time_saved:.1f}s)")


def start_vllm_server() -> subprocess.Popen:
    """Start vLLM server in background."""
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/kaggle/input/gpt-oss-120b/transformers/default/1",
        "--served-model-name", "gpt-oss",
        "--tensor-parallel-size", "1",
        "--max-num-seqs", "64",
        "--gpu-memory-utilization", "0.96",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "auto",
        "--max-model-len", str(MAX_LEN),
        "--stream-interval", "20",
    ]
    with open("./vllm.log", "w") as logfile:
        process = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
    print("vLLM server started. Logs: ./vllm.log")
    return process


vllm_process = start_vllm_server()


# Option A: Exact same as way-to-30 (proven 30/50 on LB)
TIR_PROMPT_SIMPLE = """You are a world-class mathematical professor and olympiad problem who is given a national/international-level math olympiad problem.
Think carefully and reason step by step and use the python tool to solve the math problem.
Finally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess."""

# Use simple version (same as way-to-30) - change to TIR_PROMPT_ENHANCED if needed
TIR_PROMPTS = [TIR_PROMPT_SIMPLE]


class HarmonyTIRInferencer:
    """Inferencer using Harmony protocol with TIR (Tool-Integrated Reasoning)."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = MAX_LEN,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        min_p: float = MIN_P,
        seed: int = SEED,
        k: int = K,
        use_budget: bool = USE_BUDGET,
        max_iter: int = 100,
        early_stop_threshold: int = 3,

    ):
        self.model_path = model_path
        self.model = "gpt-oss"
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.seed = seed
        self.k = k
        self.use_budget = use_budget
        self.max_iter = max_iter
        self.base_budget = 60 * 5.5  # 5.5 minutes
        self.budget = 370
        self.deadline = None
        self.early_stop_threshold = early_stop_threshold  # Stop when N answers match
        self.time_budget_manager = None  

        self.client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="sk-local",
            timeout=360,
        )
        self.stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def wait_server(self):
        """Wait for vLLM server to be ready."""
        for _ in range(15 * 60):
            time.sleep(1)
            try:
                print(self.client.models.list())
                return
            except Exception:
                continue
        raise RuntimeError("vLLM server failed to start")

    def get_num_samples(self) -> int:
        """Get number of samples based on budget."""
        if not self.use_budget:
            print(f"Budget disabled -> N: {self.k}")
            return self.k
        estimated = (self.budget - 190) / 90
        ret = min(self.k, math.floor(estimated))
        print(f"Budget: {self.budget} -> N: {ret}")
        return max(1, ret)

    def apply_chat_template(self, prompt: str, python_tool: PythonTool) -> list[Message]:
        """Create Harmony messages with system prompt and tools."""
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config)
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

    def format_prompts(self, problem: str) -> list[str]:
        """Format problem with TIR prompts."""
        num_samples = self.get_num_samples()
        prompts = []
        for i in range(num_samples):
            tir_prompt = TIR_PROMPTS[i % len(TIR_PROMPTS)]
            prompts.append(problem + "\n\n" + tir_prompt)
        return prompts

    def inference(self, problem: str, deadline: float) -> tuple[int, bool]:
        """Run inference on a problem. Returns (answer, early_stopped)."""
        self.deadline = deadline
        start_time = time.time()

        prompts = self.format_prompts(problem)
        responses, early_stopped = self._inference_parallel(prompts)

        duration = time.time() - start_time
        print(f"[inference] Took {duration:.2f}s (early_stopped={early_stopped})")

        answer = self.parse_responses(responses)
        return answer, early_stopped

    def single_generate_tir(self, prompt: str, stop_event: threading.Event) -> str:
        """Generate single TIR response with tool execution."""
        python_tool = None
        try:
            python_tool = PythonTool(execution_backend="jupyter")
            messages = self.apply_chat_template(prompt, python_tool)
            final_answer_found = ""

            for iteration in range(self.max_iter):
                # Check termination conditions
                if self.deadline and time.time() >= self.deadline:
                    print("⏰ Deadline reached")
                    break
                if final_answer_found:
                    break
                if stop_event and stop_event.is_set():
                    print("🛑 Stop signal received")
                    break

                # Render conversation to token IDs
                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )
                max_tokens = self.max_model_len - len(prompt_ids)
                if max_tokens < 1:
                    print("⚠️ Context full")
                    break

                token_buffer = []
                token_buffer_str = ""
                breaking = False

                # Stream generation
                stream = self.client.completions.create(
                    model=self.model,
                    prompt=prompt_ids,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                    stream=True,
                    extra_body=dict(
                        min_p=self.min_p,
                        stop_token_ids=self.stop_token_ids,
                        return_token_ids=True,
                    ),
                    timeout=360,
                )

                for chunk in stream:
                    if stop_event and stop_event.is_set():
                        breaking = True
                        break

                    token_chunk = chunk.choices[0].token_ids
                    text_chunk = chunk.choices[0].text

                    if token_chunk:
                        token_buffer.extend(token_chunk)
                        token_buffer_str += text_chunk

                    if self.deadline and time.time() >= self.deadline:
                        breaking = True
                        break

                    if len(token_buffer) > 60_000:
                        print("⚠️ Token limit")
                        breaking = True
                        break

                    # Check for boxed answer
                    if "}" in text_chunk and self.extract_boxed_text(token_buffer_str) is not None:
                        final_answer_found = token_buffer_str
                        breaking = True
                        break

                stream.close()

                if breaking:
                    break

                # Parse generated tokens into messages
                if token_buffer:
                    new_messages = encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                    messages.extend(new_messages)

                    last_message = messages[-1]

                    # Check if generation is complete
                    if last_message.channel == "final" or token_buffer[-1] == 200002:
                        break

                    # Check if model wants to call python tool
                    if last_message.recipient == "python":
                        print(f"🐍 Executing Python code...")
                        response_msgs = python_tool.process_sync_plus(last_message)
                        messages.extend(response_msgs)

            # Return final response
            if final_answer_found:
                return final_answer_found

            # Render full conversation
            return encoding.decode_utf8(
                encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    RenderConversationConfig(auto_drop_analysis=False)
                )
            )

        except Exception as e:
            print(f"Error in generation: {e}")
            return ""
        finally:
            if python_tool:
                python_tool.close()

    def _inference_parallel(self, prompts: list[str]) -> tuple[list[str], bool]:
        """Run parallel inference with early stopping. Returns (responses, early_stopped)."""
        stop_event = threading.Event()
        answers_collected = []
        raw_responses = [""] * len(prompts)
        early_stopped = False

        print(f"🚀 Sampling {len(prompts)} times (early stop at {self.early_stop_threshold} matches)...")

        executor = ThreadPoolExecutor(max_workers=self.k)
        try:
            future_to_idx = {
                executor.submit(self.single_generate_tir, p, stop_event): i
                for i, p in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_text = future.result()
                    raw_responses[idx] = result_text

                    ans = self.extract_boxed_text(result_text)
                    if ans is not None:
                        answers_collected.append(ans)
                        counts = Counter(answers_collected)
                        most_common_ans, count = counts.most_common(1)[0]

                        # Early stopping condition
                        if count >= self.early_stop_threshold:
                            print(f"🎯 Early stop! {most_common_ans} appeared {count} times")
                            stop_event.set()
                            early_stopped = True
                            break
                except Exception as e:
                    print(f"Task exception: {e}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return raw_responses, early_stopped

    def extract_boxed_text(self, text: str) -> int | None:
        """Extract answer from \\boxed{} or 'final answer is' patterns."""
        # Try \boxed{} pattern
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        # Handle potential expressions like "12345" or just numbers
                        clean_match = match.strip().replace(',', '').replace(' ', '')
                        val = int(float(clean_match[:20]))
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        # Try 'final answer is X' pattern
        pattern = r'(?i)final\s+answer\s*(?:is|:)?\s*(\d+)'
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        val = int(match)
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        return None

    def parse_responses(self, responses: list[str]) -> int:
        """Parse responses and return majority answer."""
        answers = [self.extract_boxed_text(r) for r in responses]

        # Filter out None
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            print("No valid answers found")
            return 0

        counter = Counter(valid_answers)
        print(f"Answers: {counter}")

        most_common = counter.most_common(1)[0][0]
        return most_common % 100000


time_budget_manager = DynamicTimeBudget(TOTAL_TIME, NUM_QUESTIONS, BUFFER_TIME)


inferencer = HarmonyTIRInferencer(
    "/kaggle/input/gpt-oss-120b/transformers/default/1",
    use_budget=USE_BUDGET,
    k=K,
    early_stop_threshold=3,
)


inferencer.time_budget_manager = time_budget_manager


inferencer.wait_server()


init_time = time.time()
cutoff_times = [int(x) for x in np.linspace(final_cutoff_time, init_time, 50 + 1)]
cutoff_times.pop()


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction with dynamic time budgeting."""
    global correct_count, total_count, predictions, time_budget_manager

    question_id = id_.item(0)
    question_text = question.item(0)

    print("=" * 60)
    print(f"ID: {question_id}")
    print(f"Question: {question_text[:200]}...")

    # Get dynamic deadline
    current_deadline = time_budget_manager.get_deadline_for_question()

    # Run inference
    question_start = time.time()
    answer, early_stopped = inferencer.inference(question_text, deadline=current_deadline)
    time_spent = time.time() - question_start

    # Record completion
    time_budget_manager.record_question_completion(time_spent, early_stopped)

    # Store prediction
    predictions[question_id] = answer

    # Check accuracy if ground truth available
    total_count += 1
    if question_id in ground_truth:
        gt = ground_truth[question_id]
        is_correct = (answer == gt)
        if is_correct:
            correct_count += 1
        status = "✅" if is_correct else "❌"
        print(f"Answer: {answer} | Ground Truth: {gt} | {status}")
        print(f"📊 Running Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    else:
        print(f"Answer: {answer}")

    print("=" * 60 + "\n")

    return pl.DataFrame({"id": question_id, "answer": answer})


# Load reference data and keep ground truth for accuracy calculation
df = pd.read_csv(
    "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
)

# Store ground truth answers for accuracy calculation (only in local mode)
ground_truth = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}

# Create input file without answers
df.drop("answer", axis=1, errors="ignore").to_csv("reference.csv", index=False)

# Track predictions for accuracy calculation
predictions = {}
correct_count = 0
total_count = 0


import kaggle_evaluation.aimo_3_inference_server

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(("reference.csv",))

    # Print final accuracy summary
    if ground_truth and total_count > 0:
        print("\n" + "=" * 50)
        print("📊 FINAL ACCURACY SUMMARY")
        print("=" * 50)
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {100*correct_count/total_count:.1f}%")
        print("=" * 50)

        # Show details
        print("\nDetails:")
        for qid, pred in predictions.items():
            if qid in ground_truth:
                gt = ground_truth[qid]
                status = "✅" if pred == gt else "❌"
                print(f"  {qid}: pred={pred}, gt={gt} {status}")

