"""HTTP + SSE 入口：把 agent 的流式事件推给前端（浏览器 / curl）。

启动：
    uvicorn server:app --reload

前端 POST /chat/stream，body：
    {"message": "...", "mode": "agent", "thread_id": "可选"}

thread_id 不传会自动生成新会话；流里的第一个事件是
    {"type": "session", "thread_id": "生成的id"}
客户端后续请求带上这个 thread_id 即可续同一段对话。

逐事件收到 `data: {...}\n\n`，type 有 token（逐字）/ text（非流式节点的整段文本）
/ tool_start / tool_end / error / done。
与 CLI（main.py）共享同一个 sqlite checkpoint 文件，两端的多轮记忆互通。

curl 快速验证：
    curl -N -X POST http://localhost:8000/chat/stream \
        -H 'Content-Type: application/json' \
        -d '{"message": "你好", "mode": "agent"}'
"""

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import make_async_checkpointer
from git_workspace import GitWorkspace
from main import build_by_mode

load_dotenv()

# 每个用户一个 git 分支 + worktree（文件系统隔离 + PR 提交）
REPO_DIR = Path(__file__).resolve().parent
git_ws = GitWorkspace(repo_dir=REPO_DIR, workspaces_dir=REPO_DIR / ".workspaces")


class ChatRequest(BaseModel):
    message: str
    mode: Literal["agent", "pipeline", "routed", "dynamic"] = "agent"
    thread_id: str | None = None   # None = 自动生成新会话 id


class PrRequest(BaseModel):
    thread_id: str
    title: str


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 关停时关闭 aiosqlite 连接，避免 loop 关闭后 worker 线程报错
    for cp in _checkpointers.values():
        with contextlib.suppress(Exception):
            await cp.conn.close()
    _checkpointers.clear()
    _agents.clear()


app = FastAPI(title="Coding Agent SSE", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 生产环境改成前端域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# 每个模式共享一个 checkpointer（一个 aiosqlite 连接，指向同一个 sqlite 文件）。
_checkpointers: dict[str, Any] = {}
# 每个 (mode, thread_id) 一个 agent：backend 指向该用户的 worktree。
_agents: dict[tuple[str, str], Any] = {}
_lock = asyncio.Lock()


async def get_user_agent(mode: str, thread_id: str):
    key = (mode, thread_id)
    if key in _agents:
        return _agents[key]
    async with _lock:
        if key in _agents:
            return _agents[key]
        if mode not in _checkpointers:
            _checkpointers[mode] = await make_async_checkpointer()
        # 该用户的 worktree（独立分支），agent 在它里面读写文件，互不干扰
        workspace = await asyncio.to_thread(git_ws.ensure, thread_id)
        _agents[key] = build_by_mode(mode, str(workspace), checkpointer=_checkpointers[mode])
        return _agents[key]


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _compact(value, limit: int = 500) -> str:
    """把工具入参压成 JSON 字符串并截断，避免整文件内容撑爆 SSE。"""
    s = json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…[截断]"


async def _stream_agent(agent, message: str, thread_id: str) -> AsyncIterator[str]:
    # 先把会话 id 发给客户端，客户端后续请求带这个 thread_id 即可续聊
    yield _sse({"type": "session", "thread_id": thread_id})

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    agen = agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    )

    # 队列 + 超时注入 keepalive：长工具调用（跑测试 / 装依赖）期间没有 token，
    # 定时发一条 ": ping" 注释，浏览器才不至于以为流断了。
    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in agen:
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001
            await queue.put(("__error__", exc))
        finally:
            await queue.put(None)

    pump_task = asyncio.create_task(pump())
    streamed_runs: set[str] = set()   # 已流式吐过 token 的模型调用，避免重复
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue

            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                yield _sse({"type": "error", "message": str(item[1])})
                break

            kind = item.get("event")
            if kind == "on_chat_model_stream":
                text = _chunk_text(item["data"]["chunk"])
                if text:
                    streamed_runs.add(item.get("run_id"))
                    yield _sse({"type": "token", "content": text})
            elif kind == "on_chat_model_end":
                # 兜底：节点内部用 .invoke() 时没有 token 流，把整段文本一次性发出去
                if item.get("run_id") not in streamed_runs:
                    text = _chunk_text(item["data"].get("output"))
                    if text:
                        yield _sse({"type": "text", "content": text})
            elif kind == "on_tool_start":
                yield _sse({
                    "type": "tool_start",
                    "name": item.get("name"),
                    "input": _compact(item["data"].get("input")),
                })
            elif kind == "on_tool_end":
                yield _sse({"type": "tool_end", "name": item.get("name")})
        yield _sse({"type": "done"})
    finally:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or uuid.uuid4().hex   # 不传就开新会话
    agent = await get_user_agent(req.mode, thread_id)
    return StreamingResponse(
        _stream_agent(agent, req.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pr")
async def create_pr(req: PrRequest):
    """把该用户分支的改动提交并提 PR；有冲突则报告给用户（不自动解）。"""
    committed = await asyncio.to_thread(git_ws.commit, req.thread_id, req.title)
    if not committed:
        return {"status": "no_changes"}
    conflicted = await asyncio.to_thread(git_ws.merge_main, req.thread_id)
    if conflicted:
        files = await asyncio.to_thread(git_ws.conflicted_files, req.thread_id)
        return {"status": "conflict", "files": files, "message": "与 main 有冲突，请手动解决"}
    url = await asyncio.to_thread(git_ws.create_pr, req.thread_id, req.title)
    return {"status": "ok", "pr_url": url}


@app.get("/health")
async def health():
    return {"status": "ok"}
