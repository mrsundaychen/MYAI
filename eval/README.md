# 评测框架

对 coding agent 做客观、可复现的评测。

## 目录结构

```
eval/
  tasks.py            # 代码生成评测集（10 题，函数级 + 单测）
  codegen_bench.py    # 判分脚本：读题 → 生成 → 跑测试 → 算 pass@1
  README.md
```

## 用法

```bash
# 验证框架（用必错的 mock 生成器，不调模型）
python3 eval/codegen_bench.py --mock

# 真实跑（DeepSeek）
python3 eval/codegen_bench.py

# 只跑前 3 题
python3 eval/codegen_bench.py --limit 3
```

## 指标

- **pass@1** = 通过单测的题目数 / 总题目数。
- 每个任务会记录 token（后续扩展）。

## 设计原则

1. **判定客观可自动跑**：用单测，不靠人工看。
2. **数据与代码分离**：`tasks.py` 只放题目（prompt + tests），评测脚本统一读。
3. **生成器可插拔**：`--mock` 用必错实现验证框架；默认走真实模型。后续可换成"完整 agent（coding_agent）"而不仅是单函数补全。

## 后续规划

- 审查集（埋 bug + recall/precision/F1）
- Bug 修复集（SWE-bench 式，resolved rate）
- 多文件集（集成测试通过率）
- 长上下文/找针集（needle recall，测压缩）
- 每个任务记录 token/steps，算效率指标
