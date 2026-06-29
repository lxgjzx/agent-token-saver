# Claude Code Token Saver

综合工具套件，帮助减少 Claude Code 的 token 消耗。

## 功能

### 1. `prep` - 智能预处理

| 命令 | 功能 |
|------|------|
| `cts prep files <paths...>` | 处理文件，去注释、去重、智能截断 |
| `cts prep prompt` | 压缩 prompt 文本 |
| `cts prep diff <a> <b>` | 对比两个路径的处理效果 |
| `cts prep pipe` | 管道模式，适合 `\|` 与 Claude Code 配合 |
| `cts prep watch` | 监控会话，接近阈值时提醒 compact |

**核心能力：**
- 自动去除 16+ 种语言的注释
- 内容去重（按 hash），避免重复读取
- 大文件智能截断（保留头尾）
- Python docstring 可选去除
- 实时显示压缩比例

### 2. `sessions` - 会话管理

| 命令 | 功能 |
|------|------|
| `cts sessions list` | 列出所有会话 |
| `cts sessions create <title>` | 创建带主题标签的会话 |
| `cts sessions info <id>` | 查看会话详情和 compact 历史 |
| `cts sessions compact-log` | 记录 compact 操作 |
| `cts sessions topics` | 主题分布统计 |
| `cts sessions stats` | 会话统计概览 |

**核心能力：**
- SQLite 持久化会话元数据
- 主题分类管理
- Compact 历史追踪
- 自动统计 token 消耗

### 3. `stats` - 统计分析

| 命令 | 功能 |
|------|------|
| `cts stats report` | 综合报告（概览、趋势、热力图、建议） |
| `cts stats files` | 文件读取热力图 |
| `cts stats trend` | Token 消耗趋势 |
| `cts stats suggest` | 节省建议 |
| `cts stats cost` | 费用估算 |

**核心能力：**
- 文件读取热力图（识别高消耗文件）
- Token 消耗趋势图
- 浪费自动检测
- 个性化节省建议
- 实时费用估算

## 安装

```bash
# 克隆或进入项目目录
cd claude-token-saver

# 安装依赖
pip install -e .

# 验证
cts --help
```

## 快速使用

```bash
# ── 预处理 ──

# 预览处理效果（dry-run）
cts prep files src/ --dry-run

# 处理并输出
cts prep files src/ -o content.md

# 管道模式，直接喂给 Claude Code
find src/ -name "*.py" | cts prep pipe --no-strip-comments | claude "分析这段代码"

# 压缩 prompt
echo "你的长 prompt..." | cts prep prompt --max-tokens 5000

# ── 会话管理 ──

# 创建新会话
cts sessions create "实现用户模块" --topic backend

# 查看所有会话
cts sessions list

# 按主题过滤
cts sessions list -t backend

# 记录 compact
cts sessions compact-log abc123 --tokens-before 100000 --tokens-after 30000

# 监控提醒
cts prep watch --interval 30

# ── 统计分析 ──

# 综合报告
cts stats report

# 文件热力图
cts stats files --limit 20 --min-tokens 10000

# 费用估算
cts stats cost --days 30 --model sonnet
```

## 典型工作流

```
1. 开始新任务
   $ cts sessions create "Feature X" --topic feature-x

2. 准备代码上下文
   $ cts prep files src/ -o context.md
   → 自动去注释、去重、截断大文件

3. 将精简内容喂给 Claude
   $ cat context.md | claude "基于以上代码..."

4. 会话中使用
   → Claude 返回结果后，记录 token 使用
   $ cts sessions compact-log <id> --tokens-before <n> --tokens-after <n>

5. 定期分析
   $ cts stats report
   → 查看浪费来源，调整使用习惯
```

## 配置

配置文件：`~/.claude-token-saver/config.yaml`

```yaml
model: claude-sonnet-4-20250514
auto_compact_threshold: 100000    # 自动 compact 的 token 阈值
compact_keep_ratio: 0.3            # compact 保留比例
strip_comments: true               # 默认去除注释
strip_docstrings: false            # 默认不去除 docstring
max_file_tokens: 50000             # 单文件最大 token
max_total_tokens: 200000           # 总 token 上限
```

修改配置：
```bash
cts config --set max_file_tokens=80000
cts config --show
cts config --reset
```

## 集成 Claude Code

### 方法 1：预处理管道
```bash
find . -name "*.py" -not -path "./.venv/*" | cts prep pipe | claude "分析项目结构"
```

### 方法 2：配合 /read 命令
```bash
# 先用 prep 生成精简版本
cts prep files src/ -o .claude/summary.md

# 在 Claude Code 中使用
> /read .claude/summary.md
```

### 方法 3：Hook 集成（未来）
在 `.claude/settings.json` 中添加 hook：
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read",
      "hooks": [{
        "type": "command",
        "command": "cts prep check --file \"${tool_input.file_path}\""
      }]
    }]
  }
}
```

## Token 节省效果预估

| 场景 | 原始 Token | 处理后 Token | 节省 |
|------|-----------|-------------|------|
| 100 个 Python 文件（有注释） | ~200K | ~80K | 60% |
| 50 个 JS/TS 文件 | ~150K | ~60K | 60% |
| 重复读取的大文件 | ~100K | ~20K | 80% |
| 超长 prompt | ~50K | ~15K | 70% |

## 项目结构

```
claude-token-saver/
├── claude_token_saver/
│   ├── __init__.py
│   ├── cli.py                 # 主 CLI 入口
│   ├── cli_prep.py            # prep 子命令
│   ├── cli_sessions.py        # sessions 子命令
│   ├── cli_stats.py           # stats 子命令
│   ├── cli_helpers.py         # 共享辅助函数
│   ├── config.py              # 配置管理
│   ├── prep/                  # 预处理核心逻辑
│   ├── sessions/              # 会话管理
│   ├── stats/                 # 统计分析
│   └── utils/                 # 工具函数
├── tests/                     # 测试
└── pyproject.toml
```

## TODO

- [ ] Hook 集成（Claude Code PreToolUse / PostToolUse）
- [ ] 自动从 Claude Code transcript 解析 token 使用
- [ ] 实时文件读取追踪（daemon）
- [ ] 可视化 Dashboard（终端 TUI）
- [ ] 多人/多项目 token 预算管理
- [ ] 智能文件选择（基于语义相似度，只发送相关文件）
- [ ] 增量上下文管理（只发送变化的部分）

## License

MIT
