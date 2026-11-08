"""代码生成评测：读题 → 生成 → 跑单测 → 算 pass@1。

三种生成方式（--mode）：
    mock   —— 必错实现，验证判分框架（不调模型）
    model  —— 裸 DeepSeek 模型（baseline，不是 MYAI）
    agent  —— MYAI agent（默认）：让 agent 把函数写到 test/solutions/ 下的独立文件

每个任务写独立文件（不覆盖），评测结果汇总到 test/report.md。

用法：
    python3 eval/codegen_bench.py --mode mock
    python3 eval/codegen_bench.py --mode model --limit 3
    python3 eval/codegen_bench.py --mode agent --limit 3
"""

import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

# 让脚本可以从任意 cwd 运行、或作为模块导入，都能找到同目录的 tasks、以及项目根目录的 agent
sys.path.insert(0, str(Path(__file__).resolve().parent))            # eval/ 目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # 项目根目录（agent 等）

from tasks import TASKS

# 评测工作区：项目根目录下的 test/（保留，方便事后逐题检查）
TEST_DIR = Path(__file__).resolve().parent.parent / "test"
SOLUTIONS_DIR = TEST_DIR / "solutions"


def _solution_path(i: int) -> Path:
    return SOLUTIONS_DIR / f"solution_{i:03d}.py"


def _extract_code(text: str) -> str:
    """从模型输出里提取代码（去掉可能的 markdown 代码块包裹）。"""
    text = (text or "").strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_via_model(prompt: str) -> str:
    """裸模型补全函数（baseline，不经过 MYAI 的编排/工具）。"""
    from dotenv import load_dotenv

    load_dotenv()
    from langchain_core.messages import HumanMessage

    from agent import build_model

    model = build_model()
    msg = HumanMessage(content=(
        "补全下面的 Python 函数。只输出完整函数代码（含签名），"
        "不要任何解释、不要 markdown 代码块：\n\n" + prompt
    ))
    resp = model.invoke([msg])
    return _extract_code(resp.content if isinstance(resp.content, str) else str(resp.content))


def generate_via_agent(agent, task, filename: str, thread_id: str) -> str:
    """让 MYAI agent 把函数写到 SOLUTIONS_DIR/filename，返回读到的内容。"""
    from langchain_core.messages import HumanMessage

    entry = task["entry_point"]
    prompt = task["prompt"]
    msg = HumanMessage(content=(
        f"请实现函数 {entry}，把它写到文件 /{filename}。"
        f"只实现这一个函数，不要写 import、不要写 if __name__。函数规格：\n\n{prompt}"
    ))
    agent.invoke(
        {"messages": [msg]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
    )
    sol = SOLUTIONS_DIR / filename
    if sol.exists():
        return sol.read_text(encoding="utf-8")
    return ""


def run_tests(code: str, tests: str) -> bool:
    """把生成代码和单测拼成一个临时文件并执行；全部断言通过返回 True。"""
    script = code + "\n\n" + tests + "\n\nif __name__ == '__main__':\n    test()\n    print('OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_report(results, mode: str) -> Path:
    """把评测结果汇总成 test/report.md。"""
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 代码生成评测报告",
        "",
        f"- 时间: {now}",
        f"- 模式: {mode}",
        f"- 题目数: {total}",
        f"- **pass@1: {passed}/{total} = {passed / total * 100:.1f}%**",
        "",
        "| task_id | 函数 | 结果 | 代码文件 |",
        "|---|---|---|---|",
    ]
    for i, (task, ok) in enumerate(results):
        mark = "PASS" if ok else "FAIL"
        lines.append(
            f"| {task['task_id']} | {task['entry_point']} | {mark} | "
            f"[solution_{i:03d}.py](solutions/solution_{i:03d}.py) |"
        )
    report = TEST_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="代码生成评测")
    parser.add_argument("--mode", choices=["mock", "model", "agent"], default="agent")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    args = parser.parse_args()

    tasks = TASKS[: args.limit] if args.limit else TASKS

    # 建目录（保留，不删）
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # agent 模式：建一个 MYAI agent（复用），workspace = test/solutions/
    agent = None
    if args.mode == "agent":
        from dotenv import load_dotenv

        load_dotenv()
        from agent import build_agent

        agent = build_agent(str(SOLUTIONS_DIR))

    results = []
    for i, t in enumerate(tasks):
        filename = f"solution_{i:03d}.py"
        if args.mode == "mock":
            code = t["prompt"] + "    pass\n"
            _solution_path(i).write_text(code, encoding="utf-8")
        elif args.mode == "model":
            code = generate_via_model(t["prompt"])
            _solution_path(i).write_text(code, encoding="utf-8")
        else:  # agent：文件由 agent 写入，这里读回来
            code = generate_via_agent(agent, t, filename, thread_id=f"eval-{i}")
            if not _solution_path(i).exists():
                _solution_path(i).write_text("", encoding="utf-8")

        ok = run_tests(code, t["tests"])
        results.append((t, ok))
        print(f"  {t['task_id']}: {'PASS' if ok else 'FAIL'}")

    report = write_report(results, args.mode)
    passed = sum(1 for _, ok in results if ok)
    print(f"\npass@1 = {passed}/{len(results)} = {passed / len(results) * 100:.1f}%")
    print(f"代码目录: {SOLUTIONS_DIR}")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
