"""按用户统计 token 用量，并用一个 mock 配额做上限检查。

用法：
    tracker = UsageTracker(user="alice", quota=100_000)
    tracker.check()                 # 超配额时抛 QuotaExceededError
    tracker.add(1234)               # 累计本次模型调用消耗的 token

agent.invoke(..., config={"callbacks": [UsageCallbackHandler(tracker)]})
后者会自动统计主 agent 和所有子 agent 的模型调用。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

DEFAULT_QUOTA = 100_000_000_000
DEFAULT_USAGE_FILE = Path(__file__).resolve().parent / ".usage.json"


class QuotaExceededError(RuntimeError):
    """用户 token 配额已用完。"""


class UsageTracker:
    """一个用户在本地的 token 用量，持久化到 JSON 文件。

    配额目前是 mock：默认所有用户 100k tokens，可用 --quota 或
    TOKEN_QUOTA 环境变量覆盖。
    """

    def __init__(
        self,
        user: str,
        quota: int | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.user = (user or "default").strip() or "default"

        if quota is None:
            quota = int(os.getenv("TOKEN_QUOTA", str(DEFAULT_QUOTA)))
        if quota <= 0:
            raise ValueError("quota must be greater than 0")
        self.quota = quota

        configured_path = path or os.getenv("AGENT_USAGE_FILE")
        self.path = Path(configured_path) if configured_path else DEFAULT_USAGE_FILE

        self._lock = Lock()
        self._data = self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=".usage-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
        except OSError:
            # 统计失败不应该中断对话；内存中的用量仍然有效。
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    # --- usage -----------------------------------------------------------

    def used_tokens(self) -> int:
        entry = self._data.get(self.user, {})
        try:
            return int(entry.get("used_tokens", 0))
        except (TypeError, ValueError):
            return 0

    def remaining(self) -> int:
        return max(self.quota - self.used_tokens(), 0)

    def is_exceeded(self) -> bool:
        return self.used_tokens() >= self.quota

    def check(self) -> None:
        if self.is_exceeded():
            raise QuotaExceededError(
                f"用户 {self.user!r} 的 token 配额已用完："
                f"{self.used_tokens()}/{self.quota}。"
            )

    def add(self, tokens: int) -> None:
        if tokens <= 0:
            return
        with self._lock:
            entry = self._data.setdefault(self.user, {})
            entry["used_tokens"] = self.used_tokens() + int(tokens)
            self._save()

    def summary(self) -> str:
        used = self.used_tokens()
        pct = used / self.quota * 100 if self.quota else 0
        return (
            f"用户 {self.user}: {used}/{self.quota} tokens "
            f"({pct:.1f}%，剩余 {self.remaining()})"
        )


def _extract_tokens(response: LLMResult) -> int:
    """从 LLMResult 中提取本次调用的 total_tokens。"""
    total = 0
    found = False

    for batch in getattr(response, "generations", []) or []:
        for generation in batch:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message else None
            if not usage:
                continue
            found = True
            total += int(
                usage.get("total_tokens")
                or (usage.get("input_tokens") or 0)
                + (usage.get("output_tokens") or 0)
            )

    if found:
        return total

    # 有些 provider 只把 usage 放在 llm_output 里。
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    return int(
        usage.get("total_tokens")
        or (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    )


class UsageCallbackHandler(BaseCallbackHandler):
    """统计主 agent 和子 agent 的所有模型调用。

    callbacks 会随 LangGraph 的父 config 传播到子 agent，所以只挂一次即可。
    raise_error=True 让配额检查在超限时真正中断调用。
    """

    raise_error = True

    def __init__(self, tracker: UsageTracker) -> None:
        self.tracker = tracker

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:  # noqa: ANN001
        # chat model 也会走这里（on_chat_model_start 默认回退到 on_llm_start）。
        self.tracker.check()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:  # noqa: ANN001
        self.tracker.add(_extract_tokens(response))
