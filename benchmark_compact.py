"""对比上下文压缩方案：全量摘要 vs 去文件内容后摘要，测量 token 节约。

不调用模型，只做确定性的 token 估算（count_tokens_approximately），
测量「摘要这一步喂给模型的输入」在新旧方案下差多少——这就是去文件内容省下的 token。

用法：python3 benchmark_compact.py
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import COMPACT_KEEP, COMPACT_TRIGGER, _strip_file_contents, count_tokens_approximately


def _fake_file(name: str, n_lines: int = 80) -> str:
    """生成一份假的代码文件（模拟 read_file 的返回内容）。"""
    body = "\n".join(f"def func_{i}(a, b):\n    return a + b" for i in range(n_lines))
    return f"# {name}\n\n{body}\n"


def _fake_grep(name: str, n_matches: int = 30) -> str:
    """模拟 grep 返回的匹配行。"""
    return "\n".join(f"{name}:{i*10}:    def func_{i}(...)" for i in range(n_matches))


def build_history(n_turns: int, files_per_turn: int = 2, lines_per_file: int = 80):
    """构造多轮对话历史：每轮读几个文件（read_file）+ 偶尔 grep。"""
    msgs = []
    for t in range(n_turns):
        msgs.append(HumanMessage(content=f"帮我处理第 {t} 个任务，看看相关代码"))
        for f in range(files_per_turn):
            fn = f"/module_{t}_{f}.py"
            msgs.append(AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"file_path": fn}, "id": f"tc-{t}-{f}"}],
            ))
            msgs.append(ToolMessage(
                content=_fake_file(fn, lines_per_file), tool_call_id=f"tc-{t}-{f}", name="read_file",
            ))
        # 偶尔来一次 grep
        msgs.append(AIMessage(content="", tool_calls=[{"name": "grep", "args": {"pattern": "func"}, "id": f"g-{t}"}]))
        msgs.append(ToolMessage(content=_fake_grep(f"/module_{t}.py"), tool_call_id=f"g-{t}", name="grep"))
        msgs.append(AIMessage(content=f"第 {t} 个任务处理完了，改动如下……"))
    return msgs


def measure(msgs):
    """返回旧/新方案的摘要输入 token 及节约量。"""
    old = msgs[:-COMPACT_KEEP]
    old_full = count_tokens_approximately(old)                       # 旧方案：全量喂给摘要
    old_stripped = count_tokens_approximately(_strip_file_contents(old))  # 新方案：去文件内容后喂给摘要
    saved = old_full - old_stripped
    return {
        "total": count_tokens_approximately(msgs),
        "old_input": old_full,
        "new_input": old_stripped,
        "saved": saved,
        "saved_pct": saved / old_full * 100 if old_full else 0,
    }


def main():
    print(f"COMPACT_TRIGGER={COMPACT_TRIGGER}  COMPACT_KEEP={COMPACT_KEEP}")
    print("说明：'摘要输入' = 触发压缩时，喂给摘要模型的那部分消息的 token 数。\n")
    header = f"{'场景':<32}{'旧方案输入':>10}{'新方案输入':>10}{'节约':>10}{'节约%':>8}"
    print(header)
    print("-" * len(header))
    for n_turns in (10, 20, 40, 80):
        msgs = build_history(n_turns)
        r = measure(msgs)
        name = f"{n_turns} 轮 × 每轮2个文件"
        print(f"{name:<32}{r['old_input']:>10}{r['new_input']:>10}{r['saved']:>10}{r['saved_pct']:>7.1f}%")
    print()
    # 文件内容占总历史的比重（说明它才是 token 大头）
    msgs = build_history(40)
    full = count_tokens_approximately(msgs)
    stripped = count_tokens_approximately(_strip_file_contents(msgs))
    print(f"40 轮历史：总 {full} token，其中文件内容约占 {full - stripped} token（{(full - stripped) / full * 100:.1f}%）")


if __name__ == "__main__":
    main()
