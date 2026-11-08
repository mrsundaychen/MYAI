"""每个用户一个 git 分支 + worktree：文件系统隔离 + PR 提交。

前提：
- 项目是 git 仓库，且有 GitHub remote（origin）。
- `gh` CLI 已安装并 `gh auth login`。

用法：
    ws = GitWorkspace(repo_dir=".", workspaces_dir=".workspaces")
    path = ws.ensure("alice")                 # 建分支 user/alice + worktree，返回路径
    # agent 在 path 里改代码（互不干扰，因为每个用户一个分支目录）
    ws.commit("alice", "实现功能X")            # 有改动才提交
    url = ws.create_pr("alice", "实现功能X")   # push + gh pr create

    解决冲突：
    if ws.merge_main("alice"):                # 把 main 合进来，返回是否有冲突
        files = ws.conflicted_files("alice")  # 冲突文件（带 <<<<<<< 标记）
        # agent 读这些文件、解冲突标记，再 commit + push 更新 PR
        ws.commit("alice", "解决冲突")
        ws.push("alice")
"""

import re
import subprocess
from pathlib import Path


class GitWorkspace:
    """管理每用户分支 + worktree + PR。"""

    def __init__(self, repo_dir: str | Path, workspaces_dir: str | Path = ".workspaces"):
        self.repo = Path(repo_dir).resolve()
        self.ws_root = Path(workspaces_dir).resolve()

    # ---- 基础 ----
    def _run(self, args, cwd=None, check=False):
        p = subprocess.run(
            args, cwd=cwd or self.repo, capture_output=True, text=True
        )
        if check and p.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} 失败: {p.stderr.strip()}")
        return p

    def branch_name(self, thread_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", thread_id)
        return f"user/{safe}"

    # ---- workspace ----
    def ensure(self, thread_id: str, base: str = "main") -> Path:
        """确保该用户的分支 + worktree 存在，返回 workspace 路径。"""
        self.ws_root.mkdir(parents=True, exist_ok=True)
        ws = self.ws_root / thread_id
        if ws.is_dir():
            return ws

        branch = self.branch_name(thread_id)
        # 更新 base：有 remote 就尝试 fetch origin/base；失败（无网络/SSH 未配）退回本地 base
        has_remote = self._run(["git", "remote"]).stdout.strip() != ""
        base_ref = base
        if has_remote:
            r = self._run(["git", "fetch", "origin", base])
            if r.returncode == 0:
                base_ref = f"origin/{base}"
            else:
                self._run(["git", "checkout", base], check=True)
        else:
            self._run(["git", "checkout", base], check=True)

        # 分支不存在就新建。用 git branch（不 checkout），避免占住主工作区
        if self._run(["git", "rev-parse", "--verify", branch]).returncode != 0:
            self._run(["git", "branch", branch, base_ref], check=True)

        self._run(["git", "worktree", "add", str(ws), branch], check=True)
        return ws

    # ---- 提交 / 推送 / PR ----
    def commit(self, thread_id: str, message: str = "coding agent changes") -> bool:
        """提交 workspace 里的改动；无改动返回 False。"""
        ws = self.ws_root / thread_id
        self._run(["git", "add", "-A"], cwd=ws)
        r = self._run(["git", "status", "--porcelain"], cwd=ws)
        if not r.stdout.strip():
            return False
        self._run(["git", "commit", "-m", message], cwd=ws, check=True)
        return True

    def push(self, thread_id: str) -> None:
        self._run(
            ["git", "push", "-u", "origin", self.branch_name(thread_id)],
            cwd=self.ws_root / thread_id,
            check=True,
        )

    def create_pr(self, thread_id: str, title: str) -> str:
        """push 并创建 PR，返回 PR url。"""
        self.push(thread_id)
        r = self._run(
            [
                "gh", "pr", "create",
                "--base", "main",
                "--head", self.branch_name(thread_id),
                "--title", title,
            ],
            cwd=self.ws_root / thread_id,
        )
        if r.returncode != 0:
            raise RuntimeError(f"gh pr create 失败: {r.stderr.strip()}")
        return r.stdout.strip()

    # ---- 冲突 ----
    def merge_main(self, thread_id: str) -> bool:
        """把 origin/main 合进用户分支；返回是否有冲突（True=有冲突标记待解）。"""
        ws = self.ws_root / thread_id
        self._run(["git", "fetch", "origin", "main"], cwd=ws, check=True)
        r = self._run(["git", "merge", "origin/main"], cwd=ws)
        return r.returncode != 0

    def conflicted_files(self, thread_id: str) -> list[str]:
        """返回有冲突标记的文件列表（git diff --name-only --diff-filter=U）。"""
        ws = self.ws_root / thread_id
        r = self._run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=ws)
        return [f for f in r.stdout.splitlines() if f]
