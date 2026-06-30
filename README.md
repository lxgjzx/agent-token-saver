# Agent Token Saver

<p align="center">
  <strong>一键减少 AI Coding Agent 的 Token 消耗</strong>
  <br/>
  支持 Claude Code · Codex · OpenClaw · Cursor · Aider · Continue · Windsurf
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#功能详解">功能</a> ·
  <a href="#配置">配置</a> ·
  <a href="#可移植性">可移植性</a> ·
  <a href="#架构">架构</a>
</p>

---

## 为什么需要 Agent Token Saver？

AI Coding Agent 在每次工具调用时都会把文件内容塞进上下文。一个大型项目动辄数十万 token，单次 `Read` 就可能吃掉几千 token。

**Agent Token Saver 从四个层面压缩 token 消耗：**

| 层面 | 手段 | 典型节省 |
|------|------|----------|
| 输入压缩 | 去注释、去 docstring、骨架提取、智能截断 | 40-80% |
| 去重 | 内容 hash 去重、结构去重、SimHash 近似去重 | 10-30% |
| 输出最小化 | Hook 输出截断、压缩、字段省略 | 50-90% |
| 增量上下文 | 只发送变更文件、路径缩写、目录索引 | 60-95% |

---

## 安装

### 一键安装（推荐）

```bash
pip install agent-token-saver && ats-setup
```

`ats-setup` 是全自动安装命令，零交互完成：**检测 Agent → 安装 Hooks → 启动 Daemon**。

### NPX 运行（无需安装）

```bash
npx agent-token-saver setup
```

通过 `npx` 直接运行，无需全局安装 Python 包。适合 CI/CD 或一次性使用。

### 手动安装

```bash
# 克隆项目
git clone https://github.com/your-repo/agent-token-saver.git
cd agent-token-saver

# 安装依赖
pip install -e .

# 自动检测并安装
ats setup
```

### 系统要求

- Python >= 3.10
- Node.js >= 16.0（仅 npx 模式需要）
- Windows 10+ / macOS 10.15+ / Linux (any modern distro)
- 无额外系统依赖（纯 Python 标准库 + 少量常用包）

---

## 快速开始

### 新用户（零配置）

```bash
# 一步完成全部初始化
ats-setup
```

这会自动检测你当前使用的 AI Coding Agent，安装对应的 Hook，并启动后台监控 Daemon。

### 验证安装

```bash
# 检查状态
ats doctor

# 测试 Hook
ats hooks test -t Read
```

### 常用工作流

```bash
# 压缩文件上下文后发给 Agent
ats prep files src/ -o context.md

# 管道模式（直接喂给 Agent）
find src/ -name "*.py" | ats prep pipe | claude "分析这段代码"

# 查看 Token 消耗报告
ats stats report

# 启动后台监控
ats daemon start
```

---

## 功能详解

### 1. `prep` — 智能预处理

将文件内容压缩为 Agent 友好的上下文。

```bash
ats prep files <paths...>     # 处理文件，去注释/去重/截断
ats prep prompt <text>        # 压缩 prompt 文本
ats prep diff <a> <b>         # 对比两个路径的处理效果
ats prep pipe                 # 管道模式（stdin → stdout）
```

**核心能力：**

- 自动去除 16+ 种语言的注释（保留 TODO/FIXME/HACK/NOTE）
- Python docstring 可选去除
- 内容去重（按内容 hash），避免重复读取同一文件
- 大文件智能截断（保留头尾，中间折叠）
- 空类体压缩、冗余 `pass` 去除、`__future__` import 清洗
- SimHash 近似重复检测
- 目录索引生成（渐进式披露，不读取文件内容）
- 自适应 detail_level（根据 token 预算自动选择 full/stripped/skeleton/block）

### 2. `agents` — 多 Agent 适配器管理

统一的 Agent 管理接口，支持 7+ AI Coding Agent。

```bash
ats agents list               # 列出所有 Agent 及安装状态
ats agents detect             # 自动检测当前运行的 Agent
ats agents setup              # 一键检测 + 安装 + 启动 daemon
ats agents install            # 安装 hooks
ats agents uninstall          # 移除 hooks
ats agents test -t Read       # 测试 hook handler
ats agents check              # 健康检查
```

**支持的 Agent：**

| Agent | 环境变量 | 配置文件 |
|-------|----------|----------|
| Claude Code | `CLAUDE_CODE` | `~/.claude/settings.local.json` |
| Codex CLI | `OPENAI_CODEX` | `~/.codex/config.json` |
| OpenClaw | `OPENCLAW` | `~/.openclaw/settings.json` |
| Cursor | `CURSOR` | `~/.cursor/settings.json` |
| Continue | `CONTINUE` | `~/.continue/config.json` |
| Windsurf | `WINDSURF` | `~/.windsurf/settings.json` |
| Aider | `AIDER` | `~/.aider/configuration.yml` |

### 3. `hooks` — Hook 管理

直接管理 Agent Hook 配置（与 `agents` 命令功能重叠，保留兼容）。

```bash
ats hooks install             # 自动检测并安装
ats hooks uninstall           # 自动检测并移除
ats hooks status              # 检查所有 Agent 的 hooks 状态
ats hooks test -t Read        # 测试 hook 是否生效
ats hooks test -t Glob --stdin  # 从 stdin 读取事件测试
```

### 4. `sessions` — 会话管理

追踪和管理 Agent 对话会话，支持自动 compact。

```bash
ats sessions list             # 列出所有会话
ats sessions create <title>   # 创建带主题标签的会话
ats sessions info <id>        # 查看会话详情和 compact 历史
ats sessions compact-log      # 记录 compact 操作
ats sessions topics           # 主题分布统计
ats sessions stats            # 会话统计概览
```

### 5. `stats` — 统计分析

全面分析 Token 消耗，识别浪费来源。

```bash
ats stats report              # 综合报告
ats stats files --limit 20    # 文件读取热力图（Top 20）
ats stats trend               # Token 消耗趋势图
ats stats suggest             # 节省建议
ats stats cost --days 30      # 费用估算（最近 30 天）
```

### 6. `transcript` — Transcript 解析

增量解析 Agent 的 transcript JSONL 文件，提取 usage 和 tool_use 事件。

```bash
ats transcript scan           # 扫描所有 Agent 的 transcript
ats transcript parse <path>   # 解析指定文件
ats transcript stats          # Token 消耗历史
ats transcript tools          # 工具调用统计
```

### 7. `daemon` — 后台监控服务

后台守护进程，定期扫描 transcript 文件，写入 analytics DB，提供 HTTP API。

```bash
ats daemon start              # 启动后台服务
ats daemon stop               # 停止服务
ats daemon status             # 查看运行状态
```

**功能：**

- 增量扫描（基于文件偏移量 + mtime，不重复处理已读内容）
- 批量写入数据库（单事务，减少 I/O）
- 周期性告警（大文件检测、异常消耗提醒）
- HTTP API（`/status` `/sessions` `/alerts`），带 Bearer Token 认证

### 8. `tui` — 实时仪表盘

终端实时显示 Token 消耗仪表盘。

```bash
ats tui                       # 启动 TUI 仪表盘
```

---

## 配置

配置文件：`~/.agent-token-saver/config.yaml`

```yaml
# 模型配置
model: claude-sonnet-4-20250514

# 自动 compact 阈值（token 数）
auto_compact_threshold: 100000

# Compact 保留策略
compact_keep_ratio: 0.3       # 保留最近 30%
compact_keep_recent: 5        # 保留最近 N 轮完整内容

# 预处理选项
strip_comments: true          # 去除注释
strip_docstrings: false       # 去除 Python docstring
max_file_tokens: 50000        # 单文件最大 token 数
max_total_tokens: 200000      # 总 token 预算

# 默认行为
auto_detail_default: false    # 自适应 detail_level
structural_dedup_default: false  # 结构去重

# 忽略目录
ignore_dirs:
  - .git
  - .svn
  - .hg
  - __pycache__
  - node_modules
  - .venv
  - venv
  - dist
  - build
  - .idea
  - .vscode

# 忽略文件
ignore_files:
  - .DS_Store
  - Thumbs.db
```

### 修改配置

```bash
ats config --show                        # 显示当前配置
ats config --set max_file_tokens=80000   # 修改配置项
ats config --reset                       # 重置为默认配置
```

---

## 可移植性

### 环境变量覆盖

所有路径均可通过环境变量自定义：

```bash
# 自定义配置目录
export CTS_CONFIG_DIR=/path/to/config/

# 自定义 analytics 数据库
export CTS_ANALYTICS_DB=/path/to/analytics.db
```

### 便携模式

不写入 `~/.agent-token-saver/`，使用本地配置：

```bash
ats --config ./local_config.yaml prep files src/
```

### Docker 部署

```dockerfile
FROM python:3.12-slim

RUN pip install agent-token-saver

# 作为预处理服务
CMD ["ats", "prep", "pipe"]
```

### NPX 运行

```bash
# 无需安装，直接运行
npx agent-token-saver setup

# 或在 package.json scripts 中使用
npx agent-token-saver prep pipe
```

适合在已有 Node.js 环境的项目中快速使用，或作为 CI 中的一次性工具。

### CI/CD 集成

```yaml
# GitHub Actions — pip 方式
- name: Install agent-token-saver
  run: pip install agent-token-saver

- name: One-click setup
  run: ats-setup

# 或使用 npx（无需预装 Python 包）
- name: Setup via npx
  run: npx agent-token-saver setup
```

---

## 架构

### 项目结构

```
claude_token_saver/
├── __init__.py
├── cli.py                  # 主 CLI 入口（ats / ats-setup）
├── cli_prep.py             # prep 子命令
├── cli_sessions.py         # sessions 子命令
├── cli_stats.py            # stats 子命令
├── cli_hooks.py            # hooks 子命令
├── cli_agents.py           # agents 子命令
├── cli_transcript.py       # transcript 子命令
├── cli_daemon.py           # daemon 子命令
├── cli_tui.py              # tui 子命令
├── cli_helpers.py          # CLI 辅助函数
├── config.py               # 配置管理
├── agents/                 # 多 Agent 适配器
│   ├── __init__.py         # 注册表 + 事件处理
│   ├── base.py             # GenericJsonAdapter 基类
│   ├── claude_code.py      # Claude Code（特殊适配）
│   ├── codex.py            # Codex CLI
│   ├── openclaw.py         # OpenClaw
│   ├── cursor.py           # Cursor
│   ├── aider.py            # Aider（YAML 配置）
│   ├── continue.py         # Continue
│   └── windsurf.py         # Windsurf
├── prep/                   # 预处理核心（包入口）
│   └── __init__.py         # 50+ 压缩优化函数
├── hooks/                  # Hook 系统
│   ├── __init__.py         # 配置生成、安装、卸载
│   └── handler.py          # Hook 事件处理入口
├── sessions/               # 会话管理（SQLite）
│   └── __init__.py
├── stats/                  # 统计分析
│   └── __init__.py
├── transcript/             # Transcript 解析
│   └── __init__.py
├── daemon/                 # 后台监控服务
│   └── __init__.py
├── tui/                    # 终端仪表盘
│   └── __init__.py
├── utils/                  # 工具函数（token 计数、路径优化等）
│   └── __init__.py
├── budget.py               # 自适应预算分配
├── compactor.py            # 对话上下文压缩
├── compression_pipeline.py # 多阶段压缩管线
├── compressor.py           # 骨架提取、符号索引
├── conversation_diff.py    # 对话 Diff 压缩
├── hook_optimizer.py       # Hook 输出最小化
├── incremental_context.py  # 增量上下文
├── path_optimizer.py       # 路径缩写
├── progressive.py          # 渐进式披露
├── simhash_dedup.py        # SimHash 近似去重
├── token_budget.py         # Token 预算分配
├── common_dedup.py         # 常见文件去重
└── gitignore.py            # .gitignore 感知过滤

tests/                       # 测试套件（441+ 用例）
```

### 适配器架构

通用 Agent 适配器只需声明类属性，无需重写任何方法（Claude Code / Aider 等特殊 Agent 除外）：

```python
# agents/codex.py
from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register,
)

@_register
class _Codex(GenericJsonAdapter):
    agent_id = AgentID.CODEX
    name = "OpenAI Codex"
    settings_path = Path.home() / ".codex" / "config.json"
    tool_map = {"read_file": "Read", "write_file": "Write", ...}
    env_vars = {"OPENAI_CODEX": "1"}
    event_type_map = {"pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
    outbound_keys = {"allow": "allow", "message": "reason", ...}
```

`_register` 是 `base.py` 中的装饰器函数，将类注册到全局 `_ADAPTER_REGISTRY`。所有适配器模块在 `agents/__init__.py` 中被导入，触发注册。

特殊适配器（Claude Code、Aider）继承 `AgentAdapter` 并重写需要差异化的方法。

### 事件处理流程

```
Agent Hook → handler.py → 适配器.parse_inbound_event() → HookEvent
                                              ↓
                                   handle_pre_tool() / handle_post_tool()
                                              ↓
                                   HookDecision
                                              ↓
                           适配器.format_outbound_decision() → Agent 格式
```

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=claude_token_saver --cov-report=term-missing
```

### 新增 Agent 适配器

1. 在 `agents/` 下创建新文件，继承 `GenericJsonAdapter`
2. 声明类属性（`agent_id`、`name`、`settings_path` 等）
3. 用 `@_register` 装饰器注册
4. 在 `agents/__init__.py` 中导入

---

## License

The MIT License (MIT)
Copyright (c) <year> <copyright holders>
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the Software.
