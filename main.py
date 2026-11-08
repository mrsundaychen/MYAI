"""CLI 入口：多轮对话 + 按用户的 token 用量统计。"""

import argparse
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from git_workspace import GitWorkspace
from usage import QuotaExceededError, UsageCallbackHandler, UsageTracker
from trace import AgentTraceHandler

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
USAGE_COMMANDS = {"usage", "/usage"}
# 运行时可用 /mode <name> 切换，name ∈ 下面这几种
MODE_NAMES = ("agent", "pipeline", "routed", "dynamic")




def message_text(message) -> str:
    """从 AIMessage 中提取可读文本。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _stream_turn(agent, user_input: str, config: dict) -> None:
    """把 agent 的流式输出打到终端：逐字 token + 工具/委派标记。

    子 agent（task 工具）在工具节点内部同步执行、对外静默，可能几十秒无输出；
    这里用后台线程流式打印，主线程在委派期间打心跳点，避免看起来像卡死。
    """
    print("\nAgent> ", end="", flush=True)
    lock = threading.Lock()
    state = {"printed_any": False, "delegating": False, "error": None}
    done = threading.Event()
    seen_tool_ids: set[str] = set()

    def worker() -> None:
        try:
            for msg, _meta in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # print("")
                if isinstance(msg, (AIMessageChunk, AIMessage)):
                    text = message_text(msg)
                    if text:
                        #print(text, end="", flush=True)
                        with lock:
                            state["printed_any"] = True
                    # for tc in msg.tool_calls or []:
                    #     tid = tc.get("id")
                    #     if tid and tid not in seen_tool_ids:
                    #         seen_tool_ids.add(tid)
                    #         name = tc.get("name", "?")
                    #         if name == "task":
                    #             print("\n  [委派] 子 agent（执行中…）", flush=True)
                    #             with lock:
                    #                 state["delegating"] = True
                    #         else:
                    #             print(f"\n  [工具] {name}", flush=True)
                # elif isinstance(msg, ToolMessage):
                #     if msg.name == "task":
                #         with lock:
                #             state["delegating"] = False
                #         print("\n  [委派完成]", flush=True)
                #     else:
                #         print("\n  [工具返回完成]", flush=True)
        except Exception as exc:  # noqa: BLE001 - 交给主线程 re-raise
            with lock:
                state["error"] = exc
        finally:
            done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    # 主线程打心跳：委派期间每 5s 打一个点，让等待有可见进度
    while True:
        done.wait(5)
        if done.is_set():
            break
        with lock:
            delegating = state["delegating"]
        if delegating:
            print("·", end="", flush=True)

    worker_thread.join()

    with lock:
        error = state["error"]
        printed_any = state["printed_any"]

    if error is not None:
        raise error  # 交给 main() 的 try/except 统一处理（含配额/其他异常）

    if printed_any:
        print(flush=True)
    else:
        print("（没有返回内容）", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Coding Agent CLI")
    parser.add_argument(
        "project_root",
        nargs="?",
        default=None,
        help="agent 操作的根目录，默认为当前目录",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("AGENT_USER", "default"),
        help="用户标识：token 配额 + 独立 git 分支/worktree（默认 default）",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="mock token 配额，默认 100000，也可用 TOKEN_QUOTA 配置",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pipeline",
        action="store_true",
        help="固定顺序管线（coding → review），无路由",
    )
    mode.add_argument(
        "--routed",
        action="store_true",
        help="路由 + 固化顺序管线（router 分流，顺序仍由边保证）",
    )
    mode.add_argument(
        "--dynamic",
        action="store_true",
        help="动态规划例子（planner 拆任务 → N 个 worker 并行 → 汇总）",
    )
    return parser.parse_args(argv)


def build_by_mode(mode: str, project_root, checkpointer=None):
    """按模式名构建对应的 agent。延迟导入，确保 .env 已加载。"""
    from agent import build_agent, build_dynamic, build_pipeline, build_routed

    if mode == "agent":
        return build_agent(project_root, checkpointer=checkpointer)
    if mode == "pipeline":
        return build_pipeline(project_root, checkpointer=checkpointer)
    if mode == "routed":
        return build_routed(project_root, checkpointer=checkpointer)
    if mode == "dynamic":
        return build_dynamic(checkpointer=checkpointer)
    raise ValueError(f"未知模式：{mode}")


def _git_commit(root: str, message: str) -> bool:
    """在 root 目录 git add -A + commit；无改动返回 False。"""
    try:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True, text=True)
        r = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        if not r.stdout.strip():
            return False
        subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[commit 失败] {exc.stderr.strip()}")
        return False


def _git_pr(root: str, title: str) -> str | None:
    """push 当前分支 + gh pr create，返回 PR url；失败返回 None。"""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        if not branch or branch == "main":
            print("[pr] 请先切到一个 feature 分支（main 不能直接提 PR）")
            return None
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=root, check=True, capture_output=True, text=True)
        r = subprocess.run(
            ["gh", "pr", "create", "--base", "main", "--head", branch, "--title", title],
            cwd=root, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[pr 失败] {r.stderr.strip()}")
            return None
        return r.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"[pr 失败] {exc.stderr.strip()}")
        return None


def _get_workspace(project_root: str | None, user: str) -> str:
    """给每个用户一个 git 分支 + worktree；非 git 仓库则退回普通目录。"""
    root = Path(project_root or os.getcwd()).resolve()
    in_git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    ).stdout.strip() == "true"
    if not in_git:
        return str(root)
    ws = GitWorkspace(repo_dir=root, workspaces_dir=root / ".workspaces")
    return str(ws.ensure(user))


def main() -> None:
    load_dotenv()
    args = parse_args()

    # 超配额时我们在 callback 里主动抛异常来中断调用，LangChain 会先打一条
    # "Error in ... callback" 日志；这里只屏蔽这条噪音，异常仍会正常传播。
    logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.CRITICAL)

    # 初始模式：命令行参数 → 模式名；运行中可用 /mode 切换。
    mode = (
        "dynamic" if args.dynamic
        else "routed" if args.routed
        else "pipeline" if args.pipeline
        else "agent"
    )
    current_user = args.user
    # 每个用户一个分支 + worktree，agent 在该 worktree 里操作
    workspace = _get_workspace(args.project_root, current_user)
    try:
        agent = build_by_mode(mode, workspace)
    except RuntimeError as exc:
        print(f"错误：{exc}")
        raise SystemExit(1)

    tracker = UsageTracker(user=current_user, quota=args.quota)
    handler = UsageCallbackHandler(tracker)

    trace_enabled = True
    tools_logging = True

    config = {
        "configurable": {"thread_id": f"cli-{current_user}"},
        "recursion_limit": 100,
        "callbacks": [handler],
    }

    print("Coding Agent 已启动（exit /usage /mode /user /trace /clear /repomap /tools /commit /pr）")
    print(f"当前模式：{mode}  详细日志：{'开' if trace_enabled else '关'}")
    print(f"工作目录：{workspace}")
    print(tracker.summary())

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("再见。")
            break
        if user_input.lower() in USAGE_COMMANDS:
            print(tracker.summary())
            continue

        if user_input.lower() in ("repomap", "/repomap"):
            from repo_map import build_repo_map

            print(build_repo_map(workspace))
            continue

        if user_input.lower() in ("clear", "/clear"):
            # 不删旧数据：换一个新的 thread_id，当前会话从空对话开始；
            # 旧对话仍留在 sqlite，下次启动同一用户仍能续上原对话。
            config["configurable"]["thread_id"] = (
                f"{config['configurable']['thread_id']}-{uuid.uuid4().hex[:8]}"
            )
            print("已开启新对话（旧对话仍保留，重启后回到原线程）。")
            continue

        if user_input.lower() in ("trace", "/trace"):
            trace_enabled = not trace_enabled
            print(f"详细日志已{'开启' if trace_enabled else '关闭'}（打印主/子 agent 每一步模型与工具调用）")
            continue

        if user_input.lower() in ("tools", "/tools"):
            tools_logging = not tools_logging
            from agent import set_tool_list_logging

            set_tool_list_logging(tools_logging)
            print(f"工具列表打印已{'开启' if tools_logging else '关闭'}（每次模型调用前打印可见工具）")
            continue

        # /commit [message]：提交改动
        cmd, _, arg = user_input.partition(" ")
        if cmd.lower() in ("commit", "/commit"):
            message = arg.strip() or "coding agent changes"
            root = workspace
            if _git_commit(root, message):
                print("已提交改动。")
            else:
                print("没有可提交的改动。")
            continue

        # /pr [title]：push 当前分支 + 提 PR
        if cmd.lower() in ("pr", "/pr"):
            title = arg.strip() or "coding agent changes"
            root = workspace
            url = _git_pr(root, title)
            if url:
                print(f"PR 已创建: {url}")
            continue

        # /user <name>：切换用户（新分支/worktree + 新会话 + 新配额）
        tokens = user_input.split()
        if tokens and tokens[0].lower() in ("user", "/user"):
            if len(tokens) == 1:
                print(f"当前用户：{current_user}，用法：/user <名字>")
                continue
            current_user = tokens[1]
            workspace = _get_workspace(args.project_root, current_user)
            try:
                agent = build_by_mode(mode, workspace)
            except RuntimeError as exc:
                print(f"切换失败：{exc}")
                continue
            tracker = UsageTracker(user=current_user, quota=args.quota)
            handler = UsageCallbackHandler(tracker)
            config["configurable"]["thread_id"] = f"cli-{current_user}"
            print(f"已切换到用户 {current_user}（工作目录 {workspace}）。")
            continue

        # 运行时切换模式：/mode 或 /mode <name>
        tokens = user_input.split()
        if tokens and tokens[0].lower() in ("mode", "/mode"):
            if len(tokens) == 1:
                print(f"可用模式：{', '.join(MODE_NAMES)}")
                print("用法：/mode <模式名>")
                continue
            new_mode = tokens[1].lower()
            if new_mode not in MODE_NAMES:
                print(f"未知模式：{new_mode}，可选：{', '.join(MODE_NAMES)}")
                continue
            try:
                agent = build_by_mode(new_mode, workspace)
            except RuntimeError as exc:
                print(f"切换失败：{exc}")
                continue
            mode = new_mode
            print(f"已切换到 {mode} 模式。")
            continue

        # 每次用户请求前检查配额；单次请求内部由 callback 再检查一次。
        try:
            tracker.check()
        except QuotaExceededError as exc:
            print(f"\n[配额] {exc}")
            print("可以用 --quota 或 TOKEN_QUOTA 调大 mock 配额后重试。")
            continue

        # 每次请求按 trace 开关刷新 callbacks（开时挂一个全新的 handler）
        config["callbacks"] = [handler] + ([AgentTraceHandler()] if trace_enabled else [])

        try:
            _stream_turn(agent, user_input, config)
        except QuotaExceededError as exc:
            print(f"\n[配额] {exc}")
            continue
        except GraphRecursionError:
            print("\n[步数上限] 任务步骤超限已停止。请缩小范围或更具体地描述。")
            continue
        except Exception as exc:  # noqa: BLE001 - CLI 层兜底
            print(f"\n出错了：{exc}")
            continue

        print(f"\n[usage] {tracker.summary()}")


if __name__ == "__main__":
    main()
