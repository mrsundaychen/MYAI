"""组装四种形态：主 agent、固定管线、路由管线、动态规划（build_dynamic）。"""

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from langchain.agents.middleware.summarization import (
    DEFAULT_SUMMARY_PROMPT,
    count_tokens_approximately,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import Send
from pydantic import BaseModel

from prompts import (
    ANSWER_PROMPT,
    CODING_PROMPT,
    MAIN_PROMPT,
    PLANNER_PROMPT,
    REVIEW_PROMPT,
    ROUTER_PROMPT,
    SYNTH_PROMPT,
    WORKER_PROMPT,
)

from repo_map import build_focused_map

# review agent 只保留 read_file：去掉 ls/glob/grep，让它没法"探索"项目，
# 只能读任务里点名的那个文件，从工具层面减少读 import 的冲动。
READ_ONLY_TOOLS = ["read_file"]

# 主 agent 只保留 task 委派能力：摘掉所有文件/命令工具，
# 让"你不负责读写/执行"从一句 prompt 变成结构约束。
MAIN_EXCLUDED_TOOLS = frozenset({
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
})


def _readonly_review_middleware(backend):
    """给 review agent 用：只保留只读文件工具，去掉写/删/执行能力。"""
    return FilesystemMiddleware(backend=backend, tools=READ_ONLY_TOOLS)


# review 的递归上限：够读完目标文件 + 写结论（约 5 步），但兜住"无限探索"。
REVIEW_RECURSION_LIMIT = 30


class _RetryModelMiddleware(AgentMiddleware[Any, Any, Any]):
    """模型调用遇瞬时错误（断连/超时）时重试，扛住 DeepSeek 偶发断连。"""

    def __init__(self, attempts: int = 3, base_delay: float = 1.5):
        self._attempts = attempts
        self._base_delay = base_delay

    def wrap_model_call(self, request, handler):
        for i in range(self._attempts):
            try:
                return handler(request)
            except Exception as exc:  # noqa: BLE001 - 瞬时网络错误重试
                if i == self._attempts - 1:
                    raise
                time.sleep(self._base_delay * (i + 1))

    async def awrap_model_call(self, request, handler):
        for i in range(self._attempts):
            try:
                return await handler(request)
            except Exception as exc:  # noqa: BLE001
                if i == self._attempts - 1:
                    raise
                await asyncio.sleep(self._base_delay * (i + 1))


def _retry_model_middleware() -> _RetryModelMiddleware:
    """返回一个模型调用重试中间件。"""
    return _RetryModelMiddleware()


_PRINT_TOOL_LIST = False


def set_tool_list_logging(on: bool) -> None:
    """开关「打印工具列表」调试日志（/tools 命令调用）。"""
    global _PRINT_TOOL_LIST
    _PRINT_TOOL_LIST = on


class _ToolListMiddleware(AgentMiddleware[Any, Any, Any]):
    """调试用：每次模型调用前打印它看到的工具列表（默认关）。"""

    def _dump_tool(self, t) -> str:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else {}
            name = t.get("name") or fn.get("name", "?")
            desc = t.get("description") or fn.get("description", "")
            params = fn.get("parameters") or t.get("parameters")
        else:
            name = getattr(t, "name", "?")
            desc = getattr(t, "description", "")
            params = None
            if hasattr(t, "get_input_jsonschema"):
                try:
                    params = t.get_input_jsonschema()
                except Exception:  # noqa: BLE001
                    params = None
            if params is None and hasattr(t, "args"):
                params = t.args
        lines = [f"  === {name} ==="]
        lines.append(f"  description: {desc}")
        if params:
            lines.append("  parameters:")
            lines.append(json.dumps(params, ensure_ascii=False, indent=2))
        else:
            lines.append("  parameters: (无)")
        return "\n".join(lines)

    def _log(self, tools) -> None:
        if not _PRINT_TOOL_LIST:
            return
        print(f"[工具列表] 本次模型调用可见 {len(tools)} 个工具:", flush=True)
        for t in tools:
            print(self._dump_tool(t), flush=True)

    def wrap_model_call(self, request, handler):
        self._log(getattr(request, "tools", None) or [])
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._log(getattr(request, "tools", None) or [])
        return await handler(request)


def _tool_list_middleware() -> _ToolListMiddleware:
    """返回一个工具列表打印中间件。"""
    return _ToolListMiddleware()


def _repo_map_tool(backend):
    """给 review agent 一个按需查询符号地图的工具。"""
    root = getattr(backend, "cwd", None) or os.getcwd()

    @tool
    def repo_map(file_path: str) -> str:
        """返回项目里某个文件及其本地 import 文件的符号地图（函数/类签名 + 行号）。

        参数 file_path 用虚拟路径，例如 /main.py。
        用它快速了解一个文件的结构和它依赖的本地模块，不用 read_file 逐个读整文件。
        """
        return build_focused_map(root, file_path)

    return repo_map


def _build_review_agent(model, backend):
    """构建 review agent：只读工具 + 仓库地图工具 + 模型调用重试。"""
    return create_deep_agent(
        model=model,
        system_prompt=REVIEW_PROMPT,
        backend=backend,
        tools=[_repo_map_tool(backend)],
        middleware=[_readonly_review_middleware(backend), _retry_model_middleware(), _tool_list_middleware()],
    )


# 外层图的上下文压缩：messages 太长时，把老的部分压成摘要，保留最近 COMPACT_KEEP 条
COMPACT_KEEP = 10
COMPACT_TRIGGER = 4000  # token 阈值（用 count_tokens_approximately 估算）


def _estimate_tokens(messages):
    """估算 token 数（4 字符 ≈ 1 token + 每条消息固定开销，比按字符数准）。"""
    return count_tokens_approximately(messages)


# 文件类工具：其结果可重新读取，压缩时只留占位符、不保留内容
FILE_TOOLS = {"read_file", "grep", "glob", "ls"}


def _strip_file_contents(messages):
    """把文件类工具结果替换成占位符（内容可重新读取，不长期留在历史里）。"""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name in FILE_TOOLS:
            out.append(ToolMessage(
                content=f"[工具 {m.name} 的内容已省略，可重新调用读取]",
                tool_call_id=m.tool_call_id,
                name=m.name,
            ))
        else:
            out.append(m)
    return out


def _compact_node(model):
    """返回一个 compact 节点：超阈值时把老消息压成摘要，并从 state 移除老消息。

    - 文件类工具结果先「去内容化」（可重新读取，不占 token）
    - 结构信息由 repo_map 工具按需获取，不在这里重复注入
    """

    def compact(state):
        messages = state["messages"]
        if _estimate_tokens(messages) <= COMPACT_TRIGGER:
            return {}
        old = messages[:-COMPACT_KEEP]
        recent = messages[-COMPACT_KEEP:]
        if not old:
            return {}
        old = _strip_file_contents(old)
        prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(old))
        summary = model.invoke([HumanMessage(content=prompt)])
        text = summary.content if isinstance(summary.content, str) else str(summary.content)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=f"[更早对话摘要]\n{text}"),
                *recent,
            ]
        }

    compact.__name__ = "compact"
    return compact


class _CompactMiddleware(AgentMiddleware[Any, Any, Any]):
    """自定义上下文压缩中间件：逻辑与 _compact_node 一致，供 agent 模式用。

    name 设为 "SummarizationMiddleware"，以替换 deepagents 默认的 summarization 中间件。
    用 before_model 钩子回写 state，让压缩持久化（否则 state 不缩、每轮都重新压缩）。
    """

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"

    def __init__(self, model, trigger=COMPACT_TRIGGER, keep=COMPACT_KEEP):
        self._model = model
        self._trigger = trigger
        self._keep = keep

    def _compact(self, messages):
        """返回 state 更新（压缩后）或 None（无需压缩）。"""
        tokens = count_tokens_approximately(messages)
        rounds = sum(1 for m in messages if isinstance(m, HumanMessage))
        # 调试输出：观察压缩触发情况，后续可移除
        print(f"[compact] tokens={tokens} / {self._trigger}, rounds={rounds}, triggered={tokens > self._trigger}")
        if tokens <= self._trigger:
            return None
        old, recent = messages[:-self._keep], messages[-self._keep:]
        if not old:
            return None
        old = _strip_file_contents(old)
        prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(old))
        summary = self._model.invoke([HumanMessage(content=prompt)])
        text = summary.content if isinstance(summary.content, str) else str(summary.content)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=f"[更早对话摘要]\n{text}"),
                *recent,
            ]
        }

    async def _acompact(self, messages):
        if count_tokens_approximately(messages) <= self._trigger:
            return None
        old, recent = messages[:-self._keep], messages[-self._keep:]
        if not old:
            return None
        old = _strip_file_contents(old)
        prompt = DEFAULT_SUMMARY_PROMPT.format(messages=get_buffer_string(old))
        summary = await self._model.ainvoke([HumanMessage(content=prompt)])
        text = summary.content if isinstance(summary.content, str) else str(summary.content)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=f"[更早对话摘要]\n{text}"),
                *recent,
            ]
        }

    def before_model(self, state, runtime):
        return self._compact(state["messages"])

    async def abefore_model(self, state, runtime):
        return await self._acompact(state["messages"])

def build_model() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入你的 API key。"
        )
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,
    )


def checkpoint_db_path() -> str:
    """sqlite checkpoint 文件路径，默认放在代码同目录（与 .usage.json 一致）。"""
    return os.getenv(
        "AGENT_CHECKPOINT_DB",
        str(Path(__file__).resolve().parent / "checkpoints.sqlite"),
    )


def make_checkpointer() -> SqliteSaver:
    """同步 sqlite checkpointer，供 CLI 的 .invoke 使用（进程级持久化）。

    WAL + busy_timeout：降低多进程/多用户并发写时的 database is locked。
    """
    conn = sqlite3.connect(checkpoint_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return SqliteSaver(conn)


async def make_async_checkpointer() -> AsyncSqliteSaver:
    """异步 sqlite checkpointer，供服务端 .astream_events 使用。"""
    import aiosqlite  # 延迟导入：只有服务端需要

    conn = await aiosqlite.connect(checkpoint_db_path())
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    return AsyncSqliteSaver(conn)


def build_agent(project_root: str | Path | None = None, checkpointer=None):
    """创建主 agent。

    project_root 是 agent 操作的根目录，默认为当前工作目录。
    """
    root = Path(project_root or os.getenv("PROJECT_ROOT") or os.getcwd()).resolve()

    backend = LocalShellBackend(root_dir=root, virtual_mode=True)

    model = build_model()
    return create_deep_agent(
        model=model,
        system_prompt=MAIN_PROMPT,
        backend=backend,
        # 主 agent 只做意图识别和委派：隐藏所有文件/命令工具，只留 task；模型调用加重试。
        middleware=[
            _ToolExclusionMiddleware(excluded=MAIN_EXCLUDED_TOOLS),
            _retry_model_middleware(),
            _tool_list_middleware(),
            # 自定义上下文压缩（逻辑与 _compact_node 一致），替换 deepagents 默认 summarization
            _CompactMiddleware(model=model),
        ],
        subagents=[
            {
                "name": "coding_agent",
                "description": (
                    "负责阅读、修改、创建代码，并执行命令和测试。"
                    "当用户要求实现功能、修 bug、改代码或运行程序时使用。"
                ),
                "system_prompt": CODING_PROMPT,
                "middleware": [_retry_model_middleware(), _tool_list_middleware()],
            },
            {
                "name": "review_agent",
                "description": (
                    "负责只读审查代码，找出 bug、边界情况和安全问题并给出建议。"
                    "当用户要求 code review 或检查代码质量时使用。"
                ),
                # 预编译 + 绑定低 recursion_limit，从结构上防止 review 无限探索。
                "runnable": _build_review_agent(build_model(), backend).with_config(
                    {"recursion_limit": REVIEW_RECURSION_LIMIT}
                ),
            },
        ],
        checkpointer=checkpointer or make_checkpointer(),
    )


class PipelineState(TypedDict):
    """固定顺序管线的状态：节点之间靠 messages 传递上下文。"""

    messages: Annotated[list[AnyMessage], add_messages]


def _node_from_agent(agent, name: str, recursion_limit: int | None = None):
    """把一个 deepagents agent 包成 LangGraph 节点。

    节点拿当前 messages 去跑对应的 agent，只把该 agent 的最终回复写回状态，
    避免把中间的 tool call 噪音塞给下一个节点。
    """

    def node(state: PipelineState) -> dict:
        config = {"recursion_limit": recursion_limit} if recursion_limit else None
        try:
            result = agent.invoke({"messages": state["messages"]}, config=config)
        except GraphRecursionError:
            # 步数超限：不崩溃，返回优雅提示，让外层继续走完。
            return {"messages": [AIMessage(content="[步数超限] 该步骤已停止，请缩小任务范围后重试。")]}
        last_ai = next(
            (m for m in reversed(result["messages"]) if getattr(m, "type", None) == "ai"),
            None,
        )
        return {"messages": [last_ai]} if last_ai is not None else {"messages": []}

    node.__name__ = name
    return node


def build_pipeline(project_root: str | Path | None = None, checkpointer=None):
    """创建「固定顺序」的管线：coding → review。

    和 build_agent 的区别：
    - coding_agent / review_agent 是图里真正的节点，不再通过 task 工具委派；
    - 顺序由边决定（add_edge），不靠主 agent 的 prompt 推理，模型无法跳过或乱序。
    """
    root = Path(project_root or os.getenv("PROJECT_ROOT") or os.getcwd()).resolve()
    model = build_model()
    backend = LocalShellBackend(root_dir=root, virtual_mode=True)

    coding_agent = create_deep_agent(
        model=model,
        system_prompt=CODING_PROMPT,
        backend=backend,
        middleware=[_retry_model_middleware(), _tool_list_middleware()],
    )
    review_agent = _build_review_agent(model, backend)

    graph = StateGraph(PipelineState)
    graph.add_node("compact", _compact_node(model))
    graph.add_node("coding", _node_from_agent(coding_agent, "coding"))
    graph.add_node(
        "review",
        _node_from_agent(review_agent, "review", recursion_limit=REVIEW_RECURSION_LIMIT),
    )
    graph.add_edge(START, "compact")
    graph.add_edge("compact", "coding")
    graph.add_edge("coding", "review")  # ← 固定顺序在这里：review 一定接在 coding 之后
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer or make_checkpointer())


class RouteDecision(BaseModel):
    """路由节点输出的结构化结果。"""

    route: Literal["code", "review", "code_then_review", "answer"]


class RoutedState(TypedDict):
    """带路由的管线状态：route 记录路由决策，供下游条件边使用。"""

    messages: Annotated[list[AnyMessage], add_messages]
    route: str


def build_routed(project_root: str | Path | None = None, checkpointer=None):
    """创建「路由 + 固化顺序」的管线。

    router 只做意图判断（LLM + structured output），决定走哪条路；
    每条路内部的顺序仍由静态边保证，例如 code_then_review 一定是 coding → review。
    """
    root = Path(project_root or os.getenv("PROJECT_ROOT") or os.getcwd()).resolve()
    model = build_model()
    backend = LocalShellBackend(root_dir=root, virtual_mode=True)

    coding_agent = create_deep_agent(
        model=model,
        system_prompt=CODING_PROMPT,
        backend=backend,
        middleware=[_retry_model_middleware(), _tool_list_middleware()],
    )
    review_agent = _build_review_agent(model, backend)

    # 路由用结构化输出。DeepSeek 不支持 response_format(json_schema)，
    # 必须显式走 function_calling 通道，否则报 "response_format type is unavailable"。
    router_llm = model.with_structured_output(RouteDecision, method="function_calling")

    def router_node(state: RoutedState) -> dict:
        decision = router_llm.invoke(
            [SystemMessage(content=ROUTER_PROMPT), *state["messages"]]
        )
        return {"route": decision.route}

    def answer_node(state: RoutedState) -> dict:
        result = model.invoke([SystemMessage(content=ANSWER_PROMPT), *state["messages"]])
        return {"messages": [result]}

    graph = StateGraph(RoutedState)
    graph.add_node("compact", _compact_node(model))
    graph.add_node("router", router_node)
    graph.add_node("coding", _node_from_agent(coding_agent, "coding"))
    graph.add_node(
        "review",
        _node_from_agent(review_agent, "review", recursion_limit=REVIEW_RECURSION_LIMIT),
    )
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "compact")
    graph.add_edge("compact", "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "code": "coding",
            "review": "review",
            "code_then_review": "coding",
            "answer": "answer",
        },
    )
    # 顺序由边保证：选了 code_then_review 时，coding 之后一定进 review，否则结束
    graph.add_conditional_edges(
        "coding",
        lambda s: "review" if s["route"] == "code_then_review" else "end",
        {"review": "review", "end": END},
    )
    graph.add_edge("review", END)
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer or make_checkpointer())


def _invoke_with_retry(runnable, messages, attempts=3, delay=2.0):
    """带重试的模型调用。

    DeepSeek 在并发 fan-out 下会偶发断连（peer closed connection），
    重试一次通常就能救回。生产上这类瞬时网络错误都该重试。
    """
    last_exc = None
    for i in range(attempts):
        try:
            return runnable.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - 这里只针对瞬时网络错误重试
            last_exc = exc
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    raise last_exc


# 哨兵：planner 每轮开头用它清空 findings，避免跨轮累积。
# 必须是可序列化的值（checkpointer 会用 msgpack 序列化写操作），所以用字符串而非 object()。
_RESET_FINDINGS = "__reset_findings__"


def _findings_reducer(current, update):
    """findings 归并规则：普通追加；遇到哨兵则清空（每轮开头）。"""
    if update == _RESET_FINDINGS:
        return []
    return (current or []) + update


class Plan(BaseModel):
    """规划节点拆出来的子任务列表，数量在运行时决定。"""

    sub_tasks: list[str]


class WorkerState(TypedDict):
    """worker 的私有输入：Send 会为每个子任务注入一个 task。"""

    task: str
    findings: Annotated[list[str], _findings_reducer]


class DynamicState(TypedDict):
    """动态规划管线的状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    sub_tasks: list[str]                          # planner 产出
    findings: Annotated[list[str], _findings_reducer]  # N 个 worker 的结果归并


def build_dynamic(checkpointer=None):
    """创建「动态规划」的例子：planner 拆任务 → N 个 worker 并行 → 汇总。

    和 build_routed 的区别：
    - build_routed 的路径是预先枚举好的（封闭选择集）；
    - build_dynamic 的子任务数量与内容由 planner 在运行时决定（开放），
      图里并不写死有几个 worker，靠 Send 动态 fan-out。

    这是一个不碰文件、纯模型并行的最小 demo，重点演示机制。
    """
    model = build_model()
    # 同 build_routed：DeepSeek 必须走 function_calling，不能用 response_format。
    planner_llm = model.with_structured_output(Plan, method="function_calling")

    def planner_node(state: DynamicState) -> dict:
        # 传全量历史，多轮引用不丢上下文
        plan = _invoke_with_retry(
            planner_llm, [SystemMessage(content=PLANNER_PROMPT), *state["messages"]]
        )
        # 兜底：planner 返回空时，把最后一条人类消息当作单个子任务
        last_user = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        sub_tasks = plan.sub_tasks or ([last_user.content] if last_user else [])
        # 每轮开头清空上一轮的 findings，避免跨轮累积
        return {"sub_tasks": sub_tasks, "findings": _RESET_FINDINGS}

    def worker_node(state: WorkerState) -> dict:
        answer = _invoke_with_retry(
            model,
            [SystemMessage(content=WORKER_PROMPT), HumanMessage(content=state["task"])],
        )
        text = answer.content if isinstance(answer.content, str) else str(answer.content)
        return {"findings": [f"[{state['task']}] {text}"]}

    def synthesizer_node(state: DynamicState) -> dict:
        joined = "\n".join(state["findings"])
        final = _invoke_with_retry(
            model, [SystemMessage(content=SYNTH_PROMPT), HumanMessage(content=joined)]
        )
        return {"messages": [final]}

    def assign_workers(state: DynamicState):
        # 运行时才知道有几个子任务，所以这里动态生成 N 个 Send
        return [Send("worker", {"task": t}) for t in state["sub_tasks"]]

    graph = StateGraph(DynamicState)
    graph.add_node("compact", _compact_node(model))
    graph.add_node("planner", planner_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "compact")
    graph.add_edge("compact", "planner")
    graph.add_conditional_edges("planner", assign_workers, ["worker"])
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile(checkpointer=checkpointer or make_checkpointer())
