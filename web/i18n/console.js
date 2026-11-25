// /console 真机工作台词典与后端 agent_settings / operator_console 结构化文案英文转换。
import { COMMON_TEXT } from './common.js';

export const CONSOLE_I18N = {
  title: 'NI-RUA · Operator Console',
  exactText: {
    '给 ': 'Prompt for ',
    ' 的指令：空闲时开始新任务，运行中可随时插话': ': start a new task when idle, or steer while running',
    '需要控制租约；': 'Control lease required; ',
    ' 空闲且动作准入开启时可发送': ' must be idle with action admission open',
    ' 原生终端': ' Native Terminal',
    'observation_profile / 相机 / 部署文件在一个 run 内固定。新建 run = 新 run 目录 + 新 ':
      'observation_profile / cameras / deployment file are fixed within a run. New run = new run directory + new ',
    ' thread：先停止并退出当前网关（Ctrl-C），再用下面的命令启动。':
      ' thread: stop and exit the current gateway (Ctrl-C), then start with the command below.',
  },
  text: {
    ...COMMON_TEXT,
    'SO-101 真机工作台': 'SO-101 Console',
    '实时预览': 'Live preview',
    '实时预览 · 仅操作员可见': 'Live Preview · Operator Only',
    'operator-only · 不进入模型上下文': 'operator-only · never enters model context',
    'wrist 相机实时预览': 'wrist camera live preview',
    'fixed 相机实时预览': 'fixed camera live preview',
    '机械臂状态': 'Arm State',
    '电机 motor（温度 / 负载 / 电压）': 'Motors (temperature / load / voltage)',
    '运行控制': 'Run Controls',
    '中断 Codex 轮次并关闭动作准入。不需要租约。': 'Interrupt Codex turn and close action admission. No lease required.',
    '中断轮次': 'Stop Turn',
    '立刻以当前测量位置为目标保持机械臂，关闭动作准入。不需要租约。': 'Immediately hold the arm at its current measured position and close action admission. No lease required.',
    '保持': 'Hold',
    '上力矩': 'Torque On',
    '松力矩': 'Relax',
    '仅在 Codex 空闲、无动作时可用': 'Available only when Codex is idle and no action is running',
    '回到 home': 'Go Home',
    '发布新观察后重新开放动作准入': 'Publish a fresh observation and reopen action admission',
    '恢复准入': 'Resume',
    '封存上次保存以来所有动作的录像，并把各摄像头画面按采集时间对齐、左右拼成一个 MP4。仅在 Codex 空闲且没有动作执行时可用；不移动机械臂，之后的任务会继续录像。':
      'Seal recordings of all actions since the last save and stitch cameras side-by-side by capture timestamp into one MP4. Available only when Codex is idle with no active action; does not move the arm, and subsequent tasks continue recording.',
    '保存任务视频': 'Save Video',
    '保存任务视频：只录动作执行的时间段，模型思考时间不录；每次保存后重新开始录下一段。':
      'Save task video: records only active motion intervals (excluding model thinking time); starts a new segment after each save.',
    '机械臂会在重力作用下掉落。': 'The arm will drop under gravity.',
    '请先用手托住机械臂或确认下方无物体与人手。松力矩后需 torque_on + resume（含新观察）才能再次运动。':
      'Support the arm by hand or clear objects and hands underneath first. Re-enabling motion requires torque_on + resume (with a fresh observation).',
    '确认松力矩 (confirm: relax)': 'Confirm Relax (confirm: relax)',
    '存在 unknown 状态': 'Unknown state detected',
    '：机械臂实际是否停止 / 动作是否执行无法确认。请到现场确认机械臂静止、无碰撞后再勾选确认；恢复准入时会先获取一次新观察。':
      ': whether the arm actually stopped or executed the action could not be confirmed. Inspect the robot in person to verify it is stationary and collision-free before checking below; resuming will capture a fresh observation first.',
    '我已在现场检查机械臂，确认以上 unknown 项': 'I have inspected the arm in person and acknowledge the unknown items above',
    'go_home 只移动机械臂到记录的 home 关节角，': 'go_home moves only the arm to its recorded home joint angles and ',
    '不会复位桌面物体': 'does not reset tabletop objects',
    '（reset_strategy=arm_home_only_world_unchanged）。stop 会关闭动作准入；之后需 resume 再发送继续指令。':
      ' (reset_strategy=arm_home_only_world_unchanged). stop closes action admission; run resume before sending further commands.',
    '安全提示：软件急停依赖串口与工作进程正常。请务必在伸手可及处为舵机电源准备':
      'Safety notice: software E-STOP depends on a healthy serial link and worker process. Always keep a ',
    '物理电源开关': 'physical power switch',
    '，任何异常（失控、冒烟、通信中断）直接断电。':
      ' for servo power within arm’s reach and cut power immediately on any anomaly (runaway motion, smoke, lost comms).',
    'Codex 运行状态': 'Codex runtime state',
    '当前 task_plan': 'Current task_plan',
    '尚无 task_plan': 'No task_plan yet',
    '任务在你自己的 Agent（Claude Code / Codex / OpenCode）里下达；这里不发送指令。':
      'Send tasks in your own Agent (Claude Code / Codex / OpenCode); prompts are not sent from this page.',
    '直接输入，回车发送（Shift+回车换行）。运行中发送 = 插话':
      'Type here; Enter to send (Shift+Enter for newline). Sending while running steers the active turn',
    '发送到当前 thread': 'Send to current thread',
    '连接中': 'Connecting',
    '模型消息与工具调用（文本摘要）': 'Model Messages & Tool Calls (Text Summary)',
    '动作回执': 'Action receipts',
    '最近动作回执': 'Recent Action Receipts',
    '模型可见观察': 'Model-visible observation',
    '模型可见观察（最近一次 robot_observe）': 'Model-Visible Observation (Latest robot_observe)',
    'model-visible · 非实时': 'model-visible · not live',
    'VGGT / SAM 感知结果 · 最近一次正式观察': 'VGGT / SAM Perception · Latest Formal Observation',
    '设置': 'Settings',
    '(a) 运行中可调': '(a) Runtime Adjustable',
    '速度上限 speed_scale_cap（0.1–1.0，默认 0.9，只能降低部署速度）':
      'Speed cap speed_scale_cap (0.1–1.0, default 0.9; can only cap deployment speed)',
    '应用 set_runtime_limits': 'Apply set_runtime_limits',
    '后端、供应商 profile、模型、推理强度和速度在「Agent」面板顶部的 Agent 栏设置。':
      'Configure backend, provider profile, model, reasoning effort, and speed in the Agent bar at the top of the Agent panel.',
    '重试模型请求': 'Retry model request',
    '(b) 需新建 run': '(b) Requires a New Run',
    '生效配置（只读，密钥已脱敏）': 'Effective Config (Read-Only, Secrets Redacted)',
  },
};

// 网关 handoff.blockers 稳定错误码英文映射
export const HANDOFF_BLOCKER_EN = Object.freeze({
  replay_is_read_only: 'Replay is read-only; cannot switch',
  benchmark_run_owned_by_queue: 'Benchmark run is managed by queue; cannot switch on this page',
  benchmark_run: 'Benchmark run is managed by queue; cannot switch on this page',
  run_starting: 'Run is still starting',
  agent_turn_running: 'Agent must be idle (current turn is still running)',
  turn_running: 'Agent must be idle',
  action_in_progress: 'An action is currently in progress',
  program_active_or_unresolved: 'A program is running or has unresolved state',
  unknown_unacknowledged: 'Unacknowledged unknown action outcome; confirm in Resume first',
  torque_not_on: 'Torque is off or unknown; enable torque first',
  handoff_in_progress: 'Handoff already accepted; finalizing current run',
  handoff_not_available: 'Current run cannot switch (replay, benchmark, or unmanaged gateway)',
  handoff_unsupported_by_backend: 'External agent runs start from the CLI and cannot switch here',
  price_card_mismatch: 'Run price card lacks pricing for the new provider/model',
  handoff_profile_key_refused: 'Provider refused this profile’s API key (HTTP 401/403)',
  gateway_without_agent_options: 'Current gateway version does not provide backend / profile switching',
});

// 将后端 agent_settings / operator_console / reasoning_options 返回的中文说明转换为地道英文（仅在 en 模式下调用）
export function translateAgentBackendText(raw, code = null) {
  if (code && HANDOFF_BLOCKER_EN[code]) return HANDOFF_BLOCKER_EN[code];
  if (raw == null) return raw;
  const text = String(raw);
  if (!/[\u4e00-\u9fff]/.test(text)) return text;

  const exact = {
    '不发送（默认）': 'Default (not sent)',
    '默认（不发送）': 'Default (not sent)',
    'summarized（摘要）': 'summarized (summary)',
    'omitted（不返回思考文本）': 'omitted (hide reasoning text)',
    '需要新会话': 'Requires new session',
    '下一轮生效': 'Applies next turn',
    '下一次模型请求生效（轮内也生效）': 'Applies on next model request (also mid-turn)',
    '正常': 'Default',
    '模型未声明支持': 'Undeclared by model',
    '当前后端不支持': 'Unsupported by current backend',
    '供应商拒绝了该档位': 'Rejected by provider',
    '模型声明不支持在会话中途更改强度（由供应商裁决）': 'Model declares no mid-session effort updates (decided by provider)',
    '请选择 profile': 'Select a profile',
    '该 profile 还没有保存密钥': 'This profile has no saved API key yet',
    '前一项被拒绝，本项未执行': 'Skipped because an earlier setting was rejected',
    '模型名格式无效': 'Invalid model ID format',
    '模型名格式无效：字母或数字开头，只含字母、数字和 . _ : / @ + - [ ]，最多 200 字符':
      'Invalid model ID: must start with a letter or digit and contain only letters, digits, and . _ : / @ + - [ ] (max 200 chars)',
    'Codex 没有思考显示选项': 'Codex has no thinking_display option',
  };
  if (Object.hasOwn(exact, text)) return exact[text];

  let match;
  if ((match = /^(.+?) 运行时(?:未安装|不可用)(?:（(.+)）)?$/.exec(text))) {
    return `${match[1]} runtime unavailable${match[2] ? ` (${match[2]})` : ''}`;
  }
  if ((match = /^该 profile 只提供 (.+)，(.+) 需要 (.+)$/.exec(text))) {
    const offered = match[1].replaceAll('（无接口）', '(no wires)').replaceAll('、', ', ');
    const needed = match[3].replaceAll(' 或 ', ' or ');
    return `Profile only serves ${offered}; ${match[2]} requires ${needed}`;
  }
  if ((match = /^该 profile 不提供 (.+) 所需的接口$/.exec(text))) {
    return `Profile does not serve a wire supported by ${match[1]}`;
  }
  if ((match = /^该组合不提供 (.+)$/.exec(text))) {
    return `This combination does not serve ${match[1]}`;
  }
  if ((match = /^profile 缺少 (.+)$/.exec(text))) {
    return `Profile missing ${match[1]}`;
  }
  if ((match = /^(.+?):\s*profile 缺少 (.+)$/.exec(text))) {
    return `${match[1]}: profile missing ${match[2]}`;
  }
  if ((match = /^profile 文件无效（(.+?)）：(.+)$/.exec(text))) {
    return `Invalid profile file (${match[1]}): ${translateAgentBackendText(match[2])}`;
  }
  if ((match = /^没有 profile (.+)$/.exec(text)) || (match = /^未知 profile (.+)$/.exec(text))) {
    return `Unknown profile ${match[1]}`;
  }
  if ((match = /^未知后端 (.+?)(?:；可选 (.+))?$/.exec(text))) {
    return `Unknown backend ${match[1]}${match[2] ? `; valid: ${match[2]}` : ''}`;
  }
  if ((match = /^(.+?) 需要新会话：当前 run 不变，用 handoff_run 收尾并启动后继 run；本次未应用任何设置$/.exec(text))) {
    const fields = match[1].replace('推理强度', 'Effort').replace('后端', 'Backend').replace('接口', 'Wire').replace('模型', 'Model').replace('思考显示', 'Thinking display');
    return `${fields} requires a new session: current run unchanged; use handoff_run to start a successor run (no settings applied)`;
  }
  if ((match = /^当前 agent 后端(?:（(.+)）)?不支持服务档位设置$/.exec(text))) {
    return `Current agent backend${match[1] ? ` (${match[1]})` : ''} does not support service tier settings`;
  }
  if ((match = /^当前供应商不支持 (.+?)（(.+)）$/.exec(text))) {
    return `Current provider does not support ${match[1]} (${match[2] === '请求被拒绝' ? 'request rejected' : match[2]})`;
  }
  if ((match = /^供应商接受了 (.+?)，但探测请求实际以 (.+?) 档位服务（以每次响应为准）$/.exec(text))) {
    const served = match[2] === '未报告的' ? 'an unreported' : match[2];
    return `Provider accepted ${match[1]}, but probe was served at ${served} tier (actual response governs)`;
  }
  if ((match = /^供应商当前不提供模型 (.+?)（(.+?)），与档位无关；请先切换到可用模型再选 (.+)$/.exec(text))) {
    return `Provider does not serve model ${match[1]} (${match[2]}); switch to an available model before selecting ${match[3]}`;
  }
  if ((match = /^无法确认供应商是否支持 (.+?)（模型 (.+?)）：(.+)$/.exec(text))) {
    return `Could not verify provider support for ${match[1]} (model ${match[2]}): ${match[3]}`;
  }
  if ((match = /^agent_model_rejected:([^:]+): 供应商拒绝该模型(.*)$/.exec(text))) {
    return `agent_model_rejected:${match[1]}: provider rejected this model${match[2].replace('（fixture）', ' (fixture)')}`;
  }
  if ((match = /^agent_effort_unsupported:([^:]+):([^:]+): 在 (.+) 上请选择其他推理强度$/.exec(text))) {
    return `agent_effort_unsupported:${match[1]}:${match[2]}: choose another reasoning effort on ${match[3]}`;
  }
  if (text.includes('思考显示只在 anthropic_messages 上可用')) {
    return text.replace(/思考显示只在 anthropic_messages 上可用（当前 (.+)）/, 'thinking_display is only available on anthropic_messages (current: $1)');
  }
  if (text.includes('Claude Code 的思考显示与预算由 CLI 环境变量决定')) {
    return 'Claude Code thinking display and budget are controlled via CLI environment variables in backends.claude_code.env (e.g. CLAUDE_CODE_THINKING_DISPLAY_UPDATES)';
  }
  if ((match = /^(.+?)未应用，本项未执行$/.exec(text))) {
    return `Skipped because an earlier field was not applied`;
  }
  return text.replaceAll('（fixture）', ' (fixture)');
}
