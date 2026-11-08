"""打印 agent 每一步调用日志的 callback handler。

用法（只挂一次，自动传播进 task 工具里的子 agent）：
    from trace import AgentTraceHandler
    agent.invoke(input, config={"callbacks": [AgentTraceHandler()]})

注意：
- on_llm_start 的 prompts 参数在本版本是序列化字符串，别当消息遍历；
  真正的模型输入在 on_chat_model_start 的 messages 参数里（BaseMessage 对象）。
"""

import json
import time

from langchain_core.callbacks import BaseCallbackHandler


def _brief(m) -> str:
    """把一条消息压成一行摘要：类型 + 长度 + 工具调用(带参数) + 开头。

    想看完整内容就把 `[:80]` 调大或删掉。
    """
    t = getattr(m, "type", "?")
    c = getattr(m, "content", "")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    c = str(c or "")
    tcs = getattr(m, "tool_calls", None) or []
    if tcs:
        parts = []
        for tc in tcs:
            name = tc.get("name", "?")
            args = tc.get("args")
            a = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args or "")
            parts.append(f"{name}({a[:600]})")
        tag = "[" + "; ".join(parts) + "]"
    else:
        tag = ""
    head = c[:800].replace("\n", " ") + ("…" if len(c) > 800 else "")
    return f"{t}({len(c)}字){tag} {head!r}"


class AgentTraceHandler(BaseCallbackHandler):
    """打印主 agent 和所有子 agent 每一步的模型/工具调用日志。"""

    def __init__(self):
        self._depth = 0          # 0=主 agent，>=1=子 agent
        self._n_llm = 0
        self._llm_start = {}     # run_id -> 开始时间
        self._tool_start = {}    # run_id -> (开始时间, 工具名)

    def _pad(self) -> str:
        return "  " * self._depth

    # ---------- 模型调用 ----------
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # 真正的模型输入：messages 是 BaseMessage 对象列表。
        # （on_llm_start 的 prompts 是序列化字符串，不能直接读；且 on_llm_start 部分路径不触发。）
        self._n_llm += 1
        self._llm_start[kwargs["run_id"]] = time.time()
        #print(f"{self._pad()}▶ 模型#{self._n_llm} 开始 [depth={self._depth}]", flush=True)
        print(f"{self._pad()}  输入 messages:", flush=True)
        for mlist in (messages or []):
            for m in mlist:
                print(f"{self._pad()}    - {_brief(m)}", flush=True)

    def on_llm_end(self, response, **kwargs):
        rid = kwargs["run_id"]
        t0 = self._llm_start.pop(rid, None)
        dur = round(time.time() - t0, 1) if t0 else 0
        for g in (getattr(response, "generations", []) or []):
            for gen in g:
                m = getattr(gen, "message", None)
                if m is None:
                    continue
                u = getattr(m, "usage_metadata", None) or {}
                i, o = u.get("input_tokens", 0), u.get("output_tokens", 0)
                # 和输入对等：用同一个 _brief 显示输出 message
                print(f"{self._pad()}  输出 message (in={i} out={o}, {dur}s):", flush=True)
                print(f"{self._pad()}    - {_brief(m)}", flush=True)

    # ---------- 工具调用 ----------
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "?")
        self._tool_start[kwargs["run_id"]] = (time.time(), name)
        if name == "task":
            print(f"{self._pad()}├─ 委派子 agent …", flush=True)
            self._depth += 1
        else:
            print(f"{self._pad()}├─ 调用工具 {name}({input_str[:120]})", flush=True)

    def on_tool_end(self, output, **kwargs):
        item = self._tool_start.pop(kwargs["run_id"], None)
        if not item:
            return
        t0, name = item
        dur = round(time.time() - t0, 2)
        if name == "task":
            self._depth -= 1
            c = getattr(output, "content", output)
            s = c if isinstance(c, str) else str(c)
            print(f"{self._pad()}└─ 子 agent 返回 {len(s)}字 ({dur}s)", flush=True)
        else:
            c = getattr(output, "content", output)
            s = c if isinstance(c, str) else str(c)
            print(f"{self._pad()}└─ 工具返回 {len(s)}字 ({dur}s)", flush=True)
