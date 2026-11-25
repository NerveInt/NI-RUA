// /index.html 轨迹与记录工作台静态词典及任务评审（task-review）文案映射。
import { COMMON_TEXT } from './common.js';

export const WORKBENCH_I18N = {
  title: 'NI-RUA · Trajectory & Record Workbench',
  exactText: {
    ' 轮次未知': ' turn unknown',
    ' · 原生 TUI': ' · Native TUI',
    ' · 原生终端': ' · Native Terminal',
    '正在读取 ': 'Loading ',
    ' 记录': ' records',
    ' 结构化记录 · 消息、工具、结果与差异': ' Structured Records · Messages, Tools, Results & Diffs',
  },
  text: {
    ...COMMON_TEXT,
    '轨迹与记录工作台': 'Trajectory & Record Workbench',
    '正在核对模式': 'Checking mode',
    '正在连接': 'Connecting',
    '停止运行': 'Stop Run',
    '持续会话任务与交互': 'Persistent session tasks & interactions',
    '持续会话': 'Persistent Session',
    '正在读取输入归属': 'Reading input ownership',
    '取得输入控制': 'Claim input',
    '机器人动作未知': 'Robot motion unknown',
    '向当前 thread 提交任务': 'Submit task to current thread',
    '输入下一轮任务；前一轮与动作结束后可提交': 'Enter next-turn task; available once the previous turn and action finish',
    '提交任务': 'Submit Task',
    '任务由唯一运行协调器提交': 'Tasks are submitted through the single run coordinator',
    '观察与动作阶段导航': 'Observation & action phase navigation',
    '按阶段回看': 'Phase Review',
    '任务与评分': 'Task & Scoring',
    '正在整理阶段': 'Indexing phases',
    '刷新阶段记录': 'Refresh Phases',
    '较早记录': 'Earlier Records',
    '完整原始事件流；可将 after 改为已读取的接收序号继续读取': 'Full raw event stream; change after to a seen sequence number to resume',
    '查看完整事件证据': 'Full Event Stream',
    '选择观察与动作阶段': 'Select observation & action phase',
    '机械臂画面': 'Robot camera view',
    '机械臂 · 阶段录像': 'Robot · Phase Video',
    '阶段录像视角': 'Phase video camera',
    '头部 · 主视角': 'Head · Main View',
    '当前阶段视频': 'Current phase video',
    '正在读取阶段与录像': 'Loading phases and recordings',
    '上一阶段': 'Prev Phase',
    '播放本阶段': 'Play Phase',
    '下一阶段': 'Next Phase',
    '连续播放，跳过无视频阶段': 'Continuous play (skip non-video phases)',
    '视频将与阶段一一对应': 'Video aligns strictly to recorded phase boundaries',
    '单次观察证据 · 图片与帧索引': 'Single-Observation Evidence · Still Images & Frame Index',
    '上一观察': 'Previous observation',
    '上一帧': 'Prev Frame',
    '按记录顺序选择历史观察': 'Select historical observation in recorded order',
    '下一观察': 'Next observation',
    '下一帧': 'Next Frame',
    '等待已记录观察': 'Waiting for recorded observations',
    '阶段指令与前后文': 'Phase prompt & surrounding context',
    '本阶段 · 指令、工具与前后文': 'Current Phase · Prompt, Tool & Context',
    '选择阶段后查看已保存的记录': 'Select a phase to inspect saved records',
    'Codex 阶段回放与原始终端': 'Codex phase replay & raw terminal',
    'Codex 回放方式': 'Codex replay mode',
    '原生 TUI': 'Native TUI',
    '阶段记录': 'Phase Records',
    '随所选阶段显示已保存的消息、工具输入与结果；空白思考记录保留为空。':
      'Shows saved messages, tool inputs, and results for the selected phase; empty reasoning blocks stay empty.',
    '重新读取': 'Reload',
    '尚未选择阶段': 'No phase selected',
    '定位本阶段原文': 'Locate in Terminal',
    '选择阶段时尝试匹配已保存的终端原文': 'Matches unique visible text in saved terminal output when selecting a phase',
    '只读监看 · 历史检查不会替换当前终端': 'Read-only monitor · historical inspection never replaces the active terminal',
    '下载当时的原始终端日志': 'Download original raw terminal log',
    '任务要求与评分': 'Task Requirements & Scoring',
    '正在读取官方规则与本次评测证据': 'Loading official rules and evaluation evidence',
    '观察与动作事件': 'Observation & action events',
    '观察与动作 · 阶段记录': 'Observations & Actions · Phase Timeline',
    '与上方视频同步': 'Synchronized with video above',
    '原始事件与独立证据': 'Raw Events & Independent Evidence',
    '等待事件': 'Waiting for events',
    '暂无事件': 'No events yet',
    '历史证据检查器': 'Historical evidence inspector',
    '证据检查': 'Evidence Inspector',
    '原始观察、动作与关联记录': 'Raw Observation, Action & Correlated Records',
    '录像文件与精确映射': 'Video Exports & Exact Frame Mapping',
    '尚无已导出的录像；录制与观测记录分别保留': 'No exported videos yet; recordings and observations are stored separately',
    '按原生 item 聚合；原始事件独立保留': 'Aggregated by native item; raw events are retained independently',
    '内部局部场景 · 版本、几何查询、快照与对齐': 'Internal Local Scene · Versions, Geometry Query, Snapshot & Alignment',
    '正在读取局部场景能力': 'Loading local scene capabilities',
    '不可变版本': 'Immutable version',
    '快照证据锚点': 'Snapshot anchor',
    '刷新版本': 'Refresh Versions',
    '保存几何快照': 'Save Geometry Snapshot',
    '几何查询参数': 'Geometry query JSON',
    '查询所选版本': 'Query Selected Version',
    '与真实观察对齐的检查参数': 'Alignment check parameters against formal observations',
    '提交对齐检查': 'Submit Alignment Check',
    '从许可 RGB 构建几何的检查参数': 'Geometry update parameters from permitted RGB',
    '提交所选基线版本的几何更新': 'Submit Geometry Update on Selected Base',
    '运行详情': 'Run Details',
    '{"observation_ids":["保留观察 ID"],"camera_id":"head","points_m":[[0,0,1]],"pixels":[[320,240]],"correspondence_source":"manual"}':
      '{"observation_ids":["retained_obs_id"],"camera_id":"head","points_m":[[0,0,1]],"pixels":[[320,240]],"correspondence_source":"manual"}',
    '{"observation_ids":["观察 A","观察 B"],"camera_id":"head","roi":[100,100,400,300],"entity_id":"table","primitive":"plane"}':
      '{"observation_ids":["obs_a","obs_b"],"camera_id":"head","roi":[100,100,400,300],"entity_id":"table","primitive":"plane"}',
  },
};

export const FAILURE_REASON_EN = Object.freeze({
  official_startup_no_action_timeout: 'No action before startup timeout',
  official_no_action_progress_timeout: 'Action progress timeout',
  model_request_limit: 'Model request budget exhausted',
  actor_repeated_inability: 'Actor declared inability 5 times in a row',
  actor_no_progress: 'No action progress after bounded recovery attempts',
  development_wall_budget: 'Wall-clock budget exhausted',
  native_step_limit: 'Native task step limit reached',
  native_task_failure: 'Native task failure',
  native_failure: 'Native task failure',
});

const FAILURE_LABEL_ZH_TO_EN = Object.freeze({
  '启动后无动作超时': 'No action before startup timeout',
  '动作进展超时': 'Action progress timeout',
  '模型调用额度耗尽': 'Model request budget exhausted',
  '模型连续 5 次声明无法继续完成': 'Actor declared inability 5 times in a row',
  '模型在有限恢复机会后仍无动作进展': 'No action progress after bounded recovery attempts',
  '实验时间预算耗尽': 'Wall-clock budget exhausted',
  '原生任务步数耗尽': 'Native task step limit reached',
  '原生任务判定失败': 'Native task failure',
});

export function translateFailureReasonEn(failure) {
  if (!failure) return undefined;
  if (typeof failure === 'string') return FAILURE_REASON_EN[failure] || FAILURE_LABEL_ZH_TO_EN[failure] || failure;
  if (failure.code && FAILURE_REASON_EN[failure.code]) return FAILURE_REASON_EN[failure.code];
  if (failure.label && FAILURE_LABEL_ZH_TO_EN[failure.label]) return FAILURE_LABEL_ZH_TO_EN[failure.label];
  return failure.label || failure.code;
}

// 将 /v1/task-review 服务端生成的常见中文模板转换为地道英文（保留原始数值与动态内容）
export function translateTaskReviewTextEn(raw) {
  if (raw == null) return raw;
  const text = String(raw);
  if (!/[\u4e00-\u9fff]/.test(text)) return text;

  const exact = {
    '运行清单': 'Run manifest',
    '运行结论': 'Run summary',
    '原生评测记录': 'Native evaluation',
    '无独立过程分': 'No separate process score',
    '分值无法静态解析': 'Score not statically parsable',
    '此环境没有官方评分规则。': 'No official scoring rules for this environment.',
    '此环境没有官方评分器（后端能力 native_evaluator 为空），本次运行不计分，也没有可核验的任务得分；任务是否完成以归档的动作回执、观测与录像为证据，不据此推算成功或分数。':
      'This environment has no official evaluator (native_evaluator is empty); this run is unscored and has no verifiable task score. Inspect archived action receipts, observations, and videos directly.',
    '本机 RoboDojo 提交与工作区差异指纹均与归档记录一致；以下规则来自该版本源码。':
      'Local RoboDojo commit and worktree diff fingerprint match the archive; rules below are parsed from that exact source version.',
    '无法确认本机源码与本次历史运行的完整版本一致；以下是当前安装版本的规则参考，不是历史逐项评分证据。':
      'Could not verify full source-version match between local checkout and this archived run; rules below are from the currently installed source for reference only.',
    '当前安装目录没有此任务的可用源码与配置，未补写评分规则。':
      'Task source and config are unavailable in the current installation; no scoring rules were synthesized.',
    '尚未保存可用的本次评测结果。': 'No usable evaluation result has been saved yet.',
    '归档保存了评测或分数快照，但没有明确记录本次是否属于正式计分样本；按局部记录展示。':
      'Archive saved an evaluation or score snapshot without confirming formal scored-episode status; shown as a partial record.',
    '归档记录任务成功；NI-RUA 的成功分支将归一化分数记为 1。原生过程分另行保留，不替代成功判定。':
      'Archive records task success; normalized score is recorded as 1. Native process score is retained separately.',
    '归档记录任务未成功且没有原生过程分；NI-RUA 将归一化分数记为 0。':
      'Archive records task failure with no native process score; normalized score is recorded as 0.',
    '按归档原值展示分数；现有字段不足以核对该值的计算分支，不重新计算或覆盖记录。':
      'Showing archived score verbatim; saved fields are insufficient to re-verify the derivation branch.',
    '本次逐条件真假、阶段到达记录及逐项得分明细未记录，不能仅根据总分反推出每项是否完成。':
      'Per-predicate verdicts and stage progression were not persisted; individual item completion cannot be inferred from the total score.',
    '阶段分依次推进，每个仿真检查步最多前进一级；显示已到达阶段的分值，各档不相加。若过程已到 100 分但任务成功条件未通过，原生评分回退到前一档。':
      'Stage score advances sequentially (at most one stage per simulation check); reports the highest reached stage without summing stages. Reverts one stage if 100 is reached without passing full completion.',
    '按已完成条件组的数量查评分阶梯，各档不相加；触发条件满足时才检查。到达 100 分但任务成功条件未通过时，原生评分回退到前一档。':
      'Looks up the score ladder by count of satisfied predicate groups when triggered (non-additive). Reverts one step if 100 is reached without full task success.',
    '各条件组对应独立分值，已完成项相加；合计为 100 分但任务成功条件未通过时，原生评分不计最后一项。':
      'Each predicate group carries an independent score summed across satisfied items; drops the final item if the sum reaches 100 without full task success.',
    '该任务未定义分段 get_score 过程分；以原生 run_reward 的成功判定为准。NI-RUA 保存的归一化分在成功时为 1，失败且无过程分时为 0。':
      'This task defines no staged get_score process score; judged solely by native run_reward binary success (normalized score 1 on success, 0 on failure).',
    '无法从可用源码确定评分聚合方式；不根据任务名称或总分补写规则。':
      'Could not determine score aggregation mode from available source; no rules were guessed.',
    '评分值与模式按当前任务源码中的登记原样提取；评分管理器版本未核验为已审阅版本，聚合细节以其源码为准。':
      'Score values and mode extracted verbatim from current task source; reward manager version was not verified against reviewed source.',
    '满足官方 run_reward 中登记的完整成功条件。': 'Satisfy all completion predicates registered in official run_reward.',
    '完整成功条件见官方 run_reward 源码。': 'See official run_reward source code below for complete success conditions.',
    '成功条件由官方 run_reward 定义；各条件的确切参数及嵌套与/或关系见下方规则源码。':
      'Success conditions are defined by official run_reward; see source code below for exact parameters and boolean structure.',
    '所有被测机械臂回到初始末端位姿：每个位置坐标的偏差不超过 0.15 m，姿态角差不超过 20°。':
      'All evaluated robot arms return to initial EE pose (position error ≤ 0.15 m per axis, orientation error ≤ 20°).',
  };
  if (Object.hasOwn(exact, text)) return exact[text];

  let match;
  if ((match = /^评分条件组 (\d+)$/.exec(text))) return `Scoring Criterion Group ${match[1]}`;
  if ((match = /^成功条件 (\d+)$/.exec(text))) return `Completion Condition ${match[1]}`;
  if ((match = /^官方 (.+) 评分登记中的第 (\d+) 组条件；完整条件和参数见规则源码。$/.exec(text))) {
    return `Group ${match[2]} in official ${match[1]} score registration; see rule source below for full parameters.`;
  }
  if ((match = /^任务失败原因：(.+?)。任务成功状态与过程得分分别记录。$/.exec(text))) {
    return `Failure reason: ${translateFailureReasonEn(match[1])}. Task success and process score are recorded separately.`;
  }
  if ((match = /^本次运行未计入正式任务评分（(.+?)）；没有可核验的正式归一化分数；任务结果与分数可用性分别记录。局部原生评测只作为当时状态的证据。$/.exec(text))) {
    return `This run is excluded from formal scoring (${match[1]}) and has no verifiable normalized score; partial native evaluations are retained as point-in-time evidence only.`;
  }
  if ((match = /^基础设施错误：(.+)$/.exec(text))) return `Infrastructure error: ${match[1]}`;
  if ((match = /^归档归一化分 = 原生过程分 (.+?) ÷ 100 = (.+?)。$/.exec(text))) {
    return `Archived normalized score = native process score ${match[1]} ÷ 100 = ${match[2]}.`;
  }
  if ((match = /^原生过程分记录为 (.+?) \/ 100，来自 RoboDojo RewardManager；native_reward 记录为 (.+?)，与过程分是不同字段。$/.exec(text))) {
    return `Native process score is ${match[1]} / 100 (from RoboDojo RewardManager); native_reward is ${match[2]} (distinct field).`;
  }
  if ((match = /^(\d+) 个数字$/.exec(text))) return `${match[1]} digits`;
  return text;
}
