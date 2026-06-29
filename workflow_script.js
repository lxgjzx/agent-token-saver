export const meta = {
  name: 'expand-token-saver',
  description: '并行扩展 claude-token-saver：Hook/Daemon/TUI/Transcript + CLI 集成',
  phases: [
    { title: '并行构建：Hook + Daemon + TUI + Transcript' },
    { title: 'CLI 集成' },
    { title: '验证' },
  ],
}

// Phase 1: 4 个模块并行构建
phase('并行构建')
const results = await parallel([
  async () => {
    const r = await agent(
      '实现 Claude Code Hook 系统。' +
      '1. 创建目录 D:\\Claude\\claude-token-saver\\claude_token_saver\\hooks\\' +
      '2. hooks/__init__.py 包含：generate_hook_config(), install_hooks(), uninstall_hooks(), merge_json_config()' +
      '3. hooks/handler.py：从 stdin 读 JSON，处理 Read/Glob/Grep 工具，输出到 stdout，记录到 analytics DB' +
      '4. 使用标准库 json sys pathlib sqlite3。',
      { label: 'hook', agentType: 'general-purpose' }
    )
    log('Hook 系统: ' + (r ? '完成' : '失败'))
    return r
  },
  async () => {
    const r = await agent(
      '实现 Daemon 监控服务。' +
      '1. 创建目录 D:\\Claude\\claude-token-saver\\claude_token_saver\\daemon\\' +
      '2. daemon/__init__.py：TokenDaemon 类，后台线程扫描 ~/.claude/projects/ transcript JSONL' +
      '3. parse_transcript() 提取 usage 和 tool_use，写入 analytics DB' +
      '4. 轻量 HTTP API（http.server）：/status, /sessions, /alerts' +
      '5. PID 文件 ~/.claude-token-saver/daemon.pid，日志 daemon.log' +
      '6. start_daemon() / stop_daemon() / get_daemon_status()' +
      '使用标准库 threading http.server json time pathlib sqlite3 signal。',
      { label: 'daemon', agentType: 'general-purpose' }
    )
    log('Daemon: ' + (r ? '完成' : '失败'))
    return r
  },
  async () => {
    const r = await agent(
      '实现 TUI Dashboard。' +
      '1. 创建目录 D:\\Claude\\claude-token-saver\\claude_token_saver\\tui\\' +
      '2. tui/__init__.py：TokenDashboard 类，使用 rich.live rich.layout rich.panel rich.table' +
      '3. 左侧：会话列表+token进度条，中间：趋势图，右侧：浪费Top5+建议，底部：状态栏' +
      '4. 实时刷新5秒，按键 q=退出 r=刷新' +
      '5. 创建 cli_tui.py：`cts tui` 命令，支持 --interval 选项' +
      '使用 rich 库。',
      { label: 'tui', agentType: 'general-purpose' }
    )
    log('TUI Dashboard: ' + (r ? '完成' : '失败'))
    return r
  },
  async () => {
    const r = await agent(
      '实现 Transcript 解析器。' +
      '1. 创建目录 D:\\Claude\\claude-token-saver\\claude_token_saver\\transcript\\' +
      '2. transcript/__init__.py：TranscriptParser 类' +
      '   - parse_directory() 扫描 ~/.claude/projects/' +
      '   - parse_file() / parse_line() 解析 JSONL' +
      '   - 提取 Session, Turn, ToolUse, UsageRecord, CostRecord' +
      '   - import_to_db() 写入 analytics DB' +
      '3. 创建 cli_transcript.py：cts transcript scan/parse/history 子命令' +
      '使用标准库 json pathlib sqlite3 datetime。',
      { label: 'transcript', agentType: 'general-purpose' }
    )
    log('Transcript: ' + (r ? '完成' : '失败'))
    return r
  },
])

const [hookOk, daemonOk, tuiOk, transcriptOk] = results.filter(Boolean)

// Phase 2: CLI 集成（等前面完成后）
phase('CLI 集成')
const cliOk = await agent(
  '更新主 CLI 入口 cli.py，注册所有新子命令：hooks, daemon, tui, transcript。' +
  '1. 修改 D:\\Claude\\claude-token-saver\\claude_token_saver\\cli.py：' +
  '   添加 import 和 main.add_command(hooks), main.add_command(daemon), main.add_command(tui), main.add_command(transcript)' +
  '2. 创建 cli_hooks.py：install, uninstall, status, test 子命令' +
  '3. 创建 cli_daemon.py：start, stop, status, logs 子命令' +
  '4. 确认 cli_tui.py 和 cli_transcript.py 存在且正确' +
  '5. 运行 python -m pytest tests/ -v 确保测试通过' +
  '6. 运行 cts --help 确认所有子命令',
  { label: 'cli-integration', agentType: 'general-purpose' }
)
log('CLI 集成: ' + (cliOk ? '完成' : '失败'))

// Phase 3: 验证
phase('验证')
const verify = await agent(
  '在 D:\\Claude\\claude-token-saver 中验证：' +
  '1. 检查所有新文件是否存在：hooks/__init__.py, hooks/handler.py, daemon/__init__.py, tui/__init__.py, tui/cli_tui.py, transcript/__init__.py, transcript/cli_transcript.py, cli_hooks.py, cli_daemon.py' +
  '2. 运行 python -m pytest tests/ -v' +
  '3. 运行 cts --help 检查所有命令' +
  '4. 报告结果',
  { label: 'verify', agentType: 'general-purpose' }
)

return {
  hook: hookOk, daemon: daemonOk, tui: tuiOk, transcript: transcriptOk,
  cli: cliOk, verify,
  parallel: true,
}
