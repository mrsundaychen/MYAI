# MYAI — Coding Agent

基于 LangGraph + deepagents 的多形态 Coding Agent，支持 CLI 与 HTTP/SSE 两种入口，
以及多用户 git 分支隔离 + PR 提交。

## 四种模式

| 模式 | 说明 |
|---|---|
| `agent`（默认） | 主 agent 做意图识别，通过 `task` 工具委派给 coding_agent / review_agent |
| `pipeline` | 固定顺序 coding → review，顺序靠图的边保证 |
| `routed` | 结构化路由分流（answer / code / review / code_then_review） |
| `dynamic` | planner 拆任务 → N 个 worker 并行 → 汇总 |

## 快速开始

### CLI（单用户）

```bash
cp .env.example .env        # 填入 DEEPSEEK_API_KEY
pip install -r requirements.txt
python3 main.py [project_root]
```

### HTTP 服务（多用户）

```bash
uvicorn server:app --reload

curl -N -X POST http://localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好"}'
```

## CLI 命令

| 命令 | 作用 |
|---|---|
| `/mode <name>` | 切换模式（agent / pipeline / routed / dynamic） |
| `/trace` | 开关详细日志：每步模型输入/输出 + 工具调用 |
| `/tools` | 开关「打印每次模型调用看到的工具列表」 |
| `/repomap` | 打印仓库地图（目录树 + 符号） |
| `/clear` | 开新对话（旧对话保留在 sqlite，重启可续） |
| `/usage` | 查看 token 用量 |

## HTTP API

### POST /chat/stream（SSE 流式对话）

body：`{"message": "...", "mode": "agent", "thread_id": "可选"}`

- `thread_id` 不传会自动生成；流的**第一个事件**是 `{"type":"session","thread_id":"..."}`，客户端后续带上它即可续同一段对话。
- 事件类型：`token`（逐字）/ `text`（非流式节点的整段文本）/ `tool_start` / `tool_end` / `error` / `done`。

### POST /pr（提交分支并提 PR）

body：`{"thread_id": "...", "title": "..."}`

返回：
- `{"status":"ok","pr_url":"..."}` —— 成功
- `{"status":"conflict","files":[...]}` —— 与 main 有冲突，**报告给用户，不自动解**
- `{"status":"no_changes"}` —— 无改动

### GET /health

## 多用户 + 分支隔离 + PR

每个用户（thread_id）一个 git 分支 + worktree，写代码物理隔离：

```
用户请求 → ensure(thread_id)
        → 建分支 user/{thread_id} + worktree .workspaces/{thread_id}/
        → agent 在该 worktree 里读写文件（互不干扰）
POST /pr → commit → 合并 origin/main 检查冲突
        → 有冲突：返回冲突文件列表，用户手动解决
        → 无冲突：push + gh pr create
```

前提（一次性）：

```bash
git remote add origin git@github.com:你的用户名/MYAI.git
brew install gh && gh auth login
git push -u origin main
```

## 关键设计

- **主 agent 只委派**：`_ToolExclusionMiddleware` 摘掉主 agent 的文件/命令工具，只留 `task`。
- **review_agent 收敛**：只读 `read_file` + `repo_map` 工具 + `recursion_limit` 兜底，避免过度探索。
- **断连重试**：`_RetryModelMiddleware` 对 DeepSeek 偶发断连重试。
- **上下文压缩**：外层图 `_compact_node` 超阈值把旧消息压成摘要。
- **checkpoint**：SQLite（WAL + busy_timeout），多轮记忆按 `thread_id` 隔离。
- **repo-map**：`build_repo_map`（全量）/ `build_focused_map`（目标文件 + 本地 import，按需实时算）。

## 目录结构

| 文件 | 作用 |
|---|---|
| `main.py` | CLI 入口 |
| `server.py` | HTTP/SSE 入口 + 多用户 + `/pr` |
| `agent.py` | 四种形态 + 中间件 + checkpointer |
| `prompts.py` | 各 agent 的 system prompt |
| `repo_map.py` | 仓库地图（全量 + 聚焦） |
| `git_workspace.py` | 每用户分支/worktree/PR/冲突 |
| `trace.py` | `AgentTraceHandler`（`/trace` 用） |
| `usage.py` | token 用量 + 配额 |

## 已知限制

- DeepSeek 长输出（几千 token 的审查报告）偶发断连，重试可能不够，建议限制输出长度或换模型。
- 文件系统隔离靠 git worktree；多用户并发改同一仓库时，冲突仍需手动解决。
- 重度并发写 SQLite 有单写者瓶颈，需要换 Postgres checkpointer。
