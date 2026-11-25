# NI-RUA 开发与架构准则

本文档是人类开发者与编码智能体在 NI-RUA（`ni_rua`）仓库工作的权威准则。修改前须先追踪端到端完整调用链，在保留既有用户工作的前提下同步完成实现、契约回归测试与文档更新。会话中用户明确给出的授权跨回合持续有效，常规可逆修复无需重复索要确认。若根目录下存在私有配套仓库 `lab/AGENTS.md`，须一并读取遵守。

## 开始前必读

1. [README](README.md) / [简体中文说明](README.zh-CN.md)：产品目标、公开工具与标准工作流。
2. [实现架构](docs/development/architecture.md)：模块职责、状态所有权与调用路径。
3. [工具契约与插件](docs/development/tool-plugins.md) 及 [扩展指南](docs/development/extensions.md)：新增或修改工具、技能与机器人后端的标准方式。
4. 按改动范围选读 [共享协议](docs/design/shared-protocol.md)、[统一动作](docs/development/unified-actions.md)、[Actor 信息边界](docs/development/actor-information-boundary.md) 与 [验证指南](docs/development/validation.md)。历史设计文档描述长期目标，当前真实契约以工具注册表、Schema 与契约测试为准。
5. 启动、恢复或整理基准实验前，必读 [实验库](docs/development/experiment-library.md)、[实验完成回调](docs/development/experiment-completion-hook.md) 及下文“实验与评测”章节。

核查 Codex 能力时优先查阅 OpenAI 官方文档，不可达或缺细节时参考 [Codex Docs 镜像](https://www.codex-docs.com/docs)；API、Hook、任务规划与版本支持结论必须对本仓固定版本 SDK/CLI、生成 Schema 或真实调用核验。

**统一命名规范**：Python 包为 `ni_rua`，发行包为 `ni-rua-harness`，命令行工具为 `ni-rua` / `ni-rua-queue` / `ni-rua-report` / `ni-rua-audit`，持久化 schema id 为 `ni_rua.*.vN`，状态目录为 `~/.{cache,config,local/share}/ni-rua`，环境变量前缀为 `NI_RUA_*`，MCP 工具命名空间为 `mcp__ni_rua__*`，程序内客户端为 `NiRuaClient`（`ni_rua_client.py`），配置文件为 `configs/ni-rua*.json`。

**默认解释器为 Miniforge 管理的 `ni-rua` 环境。** 开发、测试、网关、评测队列及离线脚本一律使用 `mamba run -n ni-rua python`（或激活 `ni-rua` 环境后的 `python`）。严禁创建 `.venv`，严禁回退系统 Python 或 `PATH` 上未固定版本的 `codex`（实际 SDK/CLI 版本与路径须对 `ni-rua` 核验并随运行记录）。Wheel 校验临时目录仅用于验证树外安装；RoboDojo 与 OpenPI 继续使用各自独立隔离解释器。

---

## 不可逾越的架构边界

**上下文只承载必要信息。** 严禁将构建、来源或存储元数据放入 Codex 上下文窗口。每个模型可见字段必须能解释其对当前判断或下一次工具调用的操作价值，否则须从生产者与展示层同步移除。哈希摘要严禁进入模型上下文，也不得替换为冗长免责声明；目标物体引用统一使用可逆短引用（`short_id`）。计划修订、执行状态、错误诊断、测量精度与证据不确定性必须完整保留。保留原生 Codex 指令、工具目录与通用编码能力，鼓励模型在沙盒与获准观测内编写分析脚本；精简只针对 Harness 额外注入的冗余指令与非必要元数据。

- **单一物理入口。** 所有影响被观测世界的动作必须经过 `Harness.execute`。新插件必须调用 `ToolContext.execute`，严禁直连机器人 Worker、伪造回执或私自推进仿真器。交互式 Actor 与辅助程序共用同一份 `ToolSpec`（`trusted_legacy` 仅限既有受信任集成）。
- **事实与展示严格分离。** 程序与测试必须消费 `ToolResponse.kind` 与 `ToolResponse.data`，绝不解析人类可读文本、`stdout` 或 HTTP 状态码推断物理成败。物理动作执行后若图像渲染或展示失败，必须保留类型化回执并附带投递错误，严禁静默重发动作；缺少可靠回执、停止确认或步数时保持 `unknown` 或 `null`。
- **完整工具契约与可解释性红线。** 每个工具须声明输入/输出 Schema、角色、副作用、可用性与 API 版本，并在 Actor 与程序双入口校验。**凡新增工具、阶段参数或内联辅助函数（如 `scene3d_helpers`），必须在契约描述中写明每个字段的物理含义、单位（`m`、`rad`、`deg`、`s`）、默认行为、典型范围与调用示例——严禁裸参数名或靠任务提示词打补丁。** 模型文档与运行时校验须同源生成并核验原生转换后载荷；复用中央注册表与 `ToolContext`，禁止平行白名单、按名路由或隐式模块扫描。
- **受限信息源与全链路脱敏。** Actor 与辅助模型仅可读取获准的正式观测；隐藏仿真状态、评测器内部实现、原始观察者产物、凭据与开发诊断严禁进入上下文。主/辅模型、程序响应、文本产物及恢复会话的 Responses API 出口共享同一元数据脱敏器与 `short_id` 映射；原始归档保留完整核验事实，但严禁作为后门暴露给模型。
- **状态仅由所有者修改。** 幂等重放返回历史回执且不回滚活跃计划。可能推进物理状态但缺终态观测的动作使既有观测失效；重新拍照不清除 `unknown` 或 `stop` 门禁。场景实体对其引用的精确观测集合校验时钟对齐与新鲜度。
- **可追溯归档与独立计分。** 封存归档必须经 `open_read_store` 打开（有 WAL 时读私有副本，无 WAL 时不可变只读打开）；活跃运行走实时 WAL 读取器。导入运行保留原生摘要并独立记录导入元数据。原图、重建图、任务得分与完整性审计分维度独立报告，未知用例保留在固定协议分母中。

---

## 研发协作、测试防膨胀与并行安全

编辑前检索全部调用方、测试、配置与导出，追根溯源修复共性根因；不引入投机性框架、外部依赖或冗余容错。非平凡逻辑须附带能复现原缺陷的回归测试（覆盖异常、部分执行、重试、权限与旧归档兼容）。不重叠文件任务可并行委派，由集成者交叉审查与核验交付。严禁覆盖其他 worktree 未提交变更、改写无关 Git 历史或干扰在跑实验；远程同步、部署与归档迁移仅限用户明确授权，本地测试严禁隐式触发。

### 测试规范与多进程并行红线

全量 CPU 测试默认通过零依赖多进程执行器 `mamba run -n ni-rua python scripts/run_tests.py` 运行（默认自动匹配宿主机 CPU 核数、上限 32 Workers；支持 `-j <N>` / `NI_RUA_TEST_JOBS`、`-k`、`-f`、`--timeout` 及模块/目录/方法过滤，100% 兼容 `python -m unittest`；`scripts/run_tests_per_module.py` 提供 `--report` / `--baseline` 单模块单进程兼容入口）：

- **严禁补丁考古碎片与提示词子串断言。** 新测试按子系统归并入既有契约测试文件（见 [验证指南](docs/development/validation.md)）。**严禁按迭代轮次新建 `test_u2_*` 等补丁考古文件或几十行微碎片；严禁断言易变的提示词自然语言子串。** 测试只断言类型化输入/输出契约、持久化状态事实与副作用计数。
- **LIFO 清理顺序（防并发 `Errno 39`）。** 同时使用临时目录与后台组件（`Store`、`ProgramRunner`）时，须先以 `self.enterContext(tempfile.TemporaryDirectory())`（或最先 `addCleanup(temp_dir.cleanup)`）入栈，再注册 `store.close` / `runner.close`，确保先停后台写线程再删目录；**严禁在 `with TemporaryDirectory():` 内注册 `addCleanup(store.close)`**。
- **零死等与环境隔离。** 本地桩服务须传短轮询间隔（如 `HTTPServer.serve_forever(poll_interval=0.01)`），子进程同步用事件/文件门控替代多秒 `time.sleep`，回环监听桩须过滤共享宿主机端口探测；`AF_UNIX` 套接字遵守 Linux `108` 字节（macOS `104` 字节）路径上限，退出零残留。

---

## 实验评测、60 并发队列与防选集偏差

实验库读取可选登记文件 `<archive_root>/service/experiment-groups.json` 维护组定义、谱系与基线（见 [实验库说明](docs/development/experiment-library.md#registry-file-format)），分数、状态与录像一律取自原始运行归档。

- **比较口径先行与固定分母。** 启动前选定基线并对齐 task、variant、整数 `seed`/`layout`、预算及干预外条件（源码、模型、供应商、SDK、提示词或记忆同变须逐项注明；缺可靠基线作探索运行，配置模型与实际 served model 分别核验）。准备、未启动、取消、无效与未评分用例保留在固定分母中；补跑须为原协议严格子集，合并时按执行顺序逐场景保留首个正式得分（含 `0` 分与正式失败），严禁挑最高分。`baseline` 与 `candidate` 分组隔离，每次执行/重试使用全新 run directory，严禁搬移或改名原始目录。记录真实源码（含未提交补丁/快照）与原生事件 UTC 起止时间，审计元数据严禁注入 Actor 上下文。
- **严禁子集刷分与选集幸存者偏差（No Subset Chasing）。** 小规模诊断集（如 `Diag-8`、`Suite-16`）**仅限验证代码与管线连通性（Smoke Check）**。严禁在按单轮成败或高波动挑出的子集上多轮迭代刷分（极易追逐右尾随机噪声触发均值回归，拉回全量集后反因回合税与注意力稀释压伤非目标基本盘）。泛化收益、默认工具目录、提示词或控制策略取舍必须在固定全量集（`Full-47`）上评估，并拆分核验目标子集与非目标基本盘无跷跷板退化。
- **全卡 60 并发队列与 256 核限流。** 等待模型、原生 Worker 或 GPU 的基准必须经 `ni_rua.apps.evaluation.queue`（兼容入口 `scripts/benchmark_queue.py`）启动。默认启用**全部可见 GPU**（`eligible_gpu_indices = null`）与 **`max_parallel = 60`**（`memory_per_case_mib = 10240`，`gpu_headroom_mib = 0`）；除非用户明确要求，严禁限卡、压低 `max_parallel` 或虚增 `gpu_headroom_mib`，严禁手拆进程、串行手工跑或直起并行网关。高核宿主机派生子进程必须注入 `DEFAULT_SUBPROCESS_THREAD_ENV`（限制 `OMP/OPENBLAS/MKL/NUMEXPR/TORCH = 1`、`RAYON/TOKIO = 2`）并配合 cgroup `pids` 余量守卫防 `pthread_create EAGAIN`；仅对明确硬件瞬态显存碰撞签名（含 PyTorch 与 Warp `CUDA error 2: out of memory`）做干净新目录重试，绝不掩盖逻辑错误。
- **回调唤醒与禁轮询纪律。** 队列带 `--wake-thread`、`--wake-codex`、`--wake-enabled` 启动后立即结束当前回合并等待回调通知（上述参数唤醒 Codex 运维会话；Claude Code 或 OpenCode 运维会话按 [完成回调协议](docs/development/experiment-completion-hook.md) 各宿主对照表，将队列作为后台任务加 `--no-wake-enabled` 启动并由宿主原生完成通知唤醒）。等待期严禁轮询 `state.json`、`tail`、查 PID/GPU、`sleep`、刷网页或派监工子智能体；无用户停止指令绝不取消在跑队列。收到回调后方读取终态、`summary.json`/`evaluation.json`、清理状态与官方评测器分数，未达或残缺则如实报未知并停止。

---

## 交付、封版默认对齐与发布

按 [验证指南](docs/development/validation.md) 执行受影响检查；涉及公共边界或发布时跑通全套 CPU（`scripts/run_tests.py`）、Web（`npm run check/build --prefix web`）与 Wheel 树外校验，记录精确 revision、命令、结果与环境约束（软件测试不替代冻结原生评测证据）。

- **静态默认配置与 `effective_config.json` 一致性门禁。** 定版、合入主干或打发布 Tag 前，必须逐项核对 `configs/ni-rua.json`、`ni_rua/core/config.py` 静态默认值与实测归档 `effective_config.json`（如 `sandbox_enabled`、`verbosity`、插件默认开关与并发参数）100% 一致，**严禁靠外部 manifest 暗中兜底默认行为**。
- 提交前执行 `git diff --check` 与敏感信息检查（禁含真实密钥、运行目录、权重、大视频或机器凭据）。变更公共契约同步更新对应指南、[README](README.md) 与 [文档索引](docs/README.md)；发布核验版本号、不可变附注 Tag、分支祖先与干净工作区，严禁改写历史 Tag。

---

## 具身控制第一性原理与防坑哲学

**最高准则：消除硬编码启发式与黑盒，将完整自主权交还模型。** 历次泛化下降、跨轮回归与分数震荡，根源皆是在底层（如 `batch_motion.py`、`action_state_ledger.py`、`vggt_height_service.py`）暗塞启发式规则、魔法数字或隐藏状态机——既剥夺模型按场景自主决策权，又制造隐藏认知错位（如暗分预抓/预放、维护隐藏 `compliant_support_z` 缓存、静默改容差），并在刚体插孔、柔性叠衣、顺应套娃与原位拧螺母间引发跨任务跷跷板效应。

**动笔前强制“四问红线自检”：**
1. **是否写死？** 是否在写死魔法数字（`0.075`、`0.0055`、`+0.002`、`-0.02`）、形状/步数阈值或单题 `if-else`？若是，立即停手！
2. **是否黑盒？** 状态跃迁或实际生效的容差/护栏对模型是否不可见？若是，拆掉黑盒改为无状态透明机制！
3. **是否有完整控制权？** 模型能否通过正交公开参数（`position_tolerance_m`、`orientation_tolerance_rad`、`max_upward_dz_m`、`stall_guard_preset`、`grip_guard_preset` / `min_aperture_m`）表达物理意图？缺旋钮则在契约公开暴露，绝不删参数让后端代猜！
4. **回执是否诚实？** 显式入参是否 100% 生效零篡改？`preview`/`execute` 是否透明回报 `effective_tolerances` 与真实物理残差？

### 七条核心控制与护栏准则

1. **严禁隐式假设、后台静默覆盖与盲裁判说教。**
   - 绝不静默覆盖显式入参或删公开参数换后台猜测（**反例**：强行 `vggt_height_mode="default"` 覆盖显式 `plane_origin_m=[0,0,0.765]` 击穿 `sweep_blocks`/`plug_in_charger`；或删 `max_upward_dz_m` 换隐藏 `_is_open_jaw_pre_grasp`、`compliant_support_z` 与 `0.075 m / 0.0055 m` 魔法数）。
   - 绝不用单目噪声包围盒或动作计数器当“盲裁判”训斥阻断模型（**反例**：倾斜 2D 投影乱报长宽比警告，或写死两步零位移 `no_scene_progress_warning` 把 `fasten_screws` 原位拧螺母误判为死锁）。
2. **绝不为单题过拟合，亦绝不陷入“面向败局残差的第二代隐形过拟合”。**
   - 机制须基于普适接触物理、运动学（主值分支 $[-\pi, \pi]$、平行夹爪对称性、硬件真实零点）与透明契约，禁调全局阈值救单题（**反例**：压低 `OT` 阈值测薄物，或让 `active_arm_grip_guard` 以 `ap_after <= 0.0005` 跨移动阶段常驻致 `fold_clothes` 步步误报空抓）。
   - **反残差打补丁**：单局失败先区分**单目随机噪声**与**底层几何/运动学缺陷**。严禁针对偶发滑脱或偏斜往 Schema/提示词堆砌专用预设、败局口诀或恢复套路（会稀释全局 40+ 任务物理注意力）；观测噪声绝不改代码，底层缺陷只修通用物理不变量并在全量集证明帕累托改进。
3. **将决策与控制权彻底交还模型（选择、判断、配置、调用）。**
   - 感知与执行只客观回报几何测量、3D 差分（`object_state_delta`）、接触归因（`contact_object_3d`）、物理残差与 `effective_tolerances`；绝不替模型命名物体、指定抓取点或硬塞刚性公式（**反例**：提示词硬塞 `grasp_yaw + (target_yaw - yaw_deg)` 致斜视噪声把直立插销拧歪）。
4. **低摩擦工具：严禁“独立查询工具双回合税”与“用粗糙单目场替代视觉闭环”。**
   - 坚持“随观测零回合税内联 `object_table`（含主值分支 `across_yaw_deg`、`stand_upright_angles_deg`、真实 `pad_z_m` 与 `scene3d` 点云句柄）+ 沙盒按需 `scene3d_helpers`（`triangulate_rays`、`in_hand_axis`、`surface_extrema`）+ 按需 `obstacle_object_ids`”。
   - **严禁将粗糙单目 3D/空间场查询或姿态运算拆成高频独立主工具**（如暴露 `spatial_field_ops`、`grasp_candidates_3d`、`se3_compose`）：既强征一轮 API 往返“双回合税”耗尽预算，又诱使模型迷信粗糙单目点云而放弃 `observe + batch_motion` 亚厘米视觉闭环。
5. **严格执行阶段终端精度与参数透明。**
   - 多阶段批处理（`[approach -> grasp -> lift -> align -> insert -> release]`）中间阶段终端核验是防误差逐级放大的防火墙。按阶段暴露容差与上浮顺应参数并透明返回 `effective_tolerances`，严格按模型显式边界执法，严禁暗自放宽将受阻偏折粉饰为 `"completed"`。
6. **物理护栏对齐真实硬件零点且完全可配置。**
   - **预抓下压受阻 vs 显式柔顺放置**：未请求宽柔顺的开爪下压若顶碰物体停在上方，诚实报 `tracking_error` 并透出 `contact_block_summary`（`blocked_above_target`、`approach_shortfall_m`），绝不半空盲闭夹爪；确需贴合或插装时由模型显式设 `stall_guard_preset="contact_stop"`、`tolerance_preset="compliant"` 或 `max_upward_dz_m`。
   - **物体自适应 `grip_guard`**：硬件真实空夹零点为 `0.0000 m`，薄壁/杯沿约 `2–13 mm`，极薄织物受压近 `0.0000 m`。模型以 `grip_guard_preset`（`none`、`thin_object`、`non_empty`、`rigid_object`、`wide_object`）或 `min_aperture_m`/`max_aperture_m` 控制；闭爪后首次上提滑脱检测（`jaw_delta <= -0.08` 且 `ap_after <= 0.002 m`）最多触发一次，严禁跨后续搬运常驻误拦薄物。
7. **防“检查点税”、多机制耦合与关节缠绕。**
   - **异常驱动暂停**：长视距任务禁诱导每步插 `observe` / `checkpoint_after_gripper`（免中断翻倍耗尽预算），靠硬件级 `grip_guard`/`stall_guard` 遇真实异常自动中断。
   - **单变量消融与主值分支防缠绕**：`1 + 1 < 2` 时凭 diff 与阶段遥测隔离归因；旋转关节等价重规划优先落入 $[-\pi, \pi]$ 防越过 $\pm\pi$ 撞上 $\pm 2\pi$ 软限位；大型评测清单与并行测试按 LPT 降序派发并配多 Key 负载均衡。
