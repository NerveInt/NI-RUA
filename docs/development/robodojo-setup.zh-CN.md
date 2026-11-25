[English](robodojo-setup.md) · [简体中文](robodojo-setup.zh-CN.md)

# RoboDojo 运行环境配置

NI-RUA 自身使用单一 Python 环境（Miniforge `NI-RUA`），而机器人仿真 Worker 则运行在独立安装的 RoboDojo / Isaac Sim 解释器中，从而保证两套依赖环境彼此隔离。本指南介绍了 RoboDojo 端所需准备的工作、必须应用的补丁，以及 NI-RUA 如何定位该安装路径。关于 Harness 环境的搭建、Web 界面的构建以及首次运行启动，请参阅[运行指南](running.zh-CN.md)。

## 固定代码版本

当前经过充分验证的硬件与仿真组合为搭配双臂 ARX X5 的 RoboDojo。外部代码库检出已固定在以下版本：

| 代码库 | 相对 RoboDojo 根目录的位置 | 固定版本（Revision） |
| --- | --- | --- |
| RoboDojo | `.` | `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` |
| IsaacLab | `third_party/IsaacLab` (submodule) | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| cuRobo | `third_party/curobo` (submodule) | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| XPolicyLab | `XPolicyLab` | `bb9a0b5f5136a74503b679af830bfd0a3a837d5c` |

请按照[上游 RoboDojo 仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)在上述固定版本的安装说明配置 Isaac Sim 5.1、CUDA 以及任务资源资产，直到获得一个能够独立运行 RoboDojo 评测的 Python 环境，然后应用下文的补丁。

## 兼容性补丁

三个补丁均位于 [`docs/development/patches/`](patches/)。它们仅处理运行时兼容性问题——不包含任何任务求解作弊逻辑，亦不改变评分规则：

| 补丁文件 | 目标目录 | 用途 |
| --- | --- | --- |
| [`robodojo.patch`](patches/robodojo.patch) | RoboDojo 根目录 | `src/eval_client/main.py`：解耦 CUDA 与 Vulkan 的 GPU 编号分配。 |
| [`isaaclab.patch`](patches/isaaclab.patch) | `third_party/IsaacLab` | `app_launcher.py` 中的 GPU 编号处理，以及 `setup.py` 中 Isaac Sim 5.1 的 Starlette 版本兼容性。 |
| [`native-runtime-compatibility.patch`](patches/native-runtime-compatibility.patch) | RoboDojo 根目录 | 将环境原点的设备/数据类型与相对位姿的位姿张量对齐；在绑定 CUDA 织物视图前注册 PhysX，并保留奖励查询所需的原生位置与四元数。 |

```sh
export NI_RUA_REPO_ROOT="$PWD"                     # NI-RUA 代码库根目录
export NI_RUA_ROBODOJO_ROOT="$HOME/code/robodojo"  # RoboDojo 代码库根目录

git -C "$NI_RUA_ROBODOJO_ROOT" checkout 726e9aabfaa642203722eb126f5eaf0f37f3e1ad
git -C "$NI_RUA_ROBODOJO_ROOT" submodule update --init --recursive

for patch in robodojo native-runtime-compatibility; do
  git -C "$NI_RUA_ROBODOJO_ROOT" apply --check "$NI_RUA_REPO_ROOT/docs/development/patches/$patch.patch" &&
  git -C "$NI_RUA_ROBODOJO_ROOT" apply "$NI_RUA_REPO_ROOT/docs/development/patches/$patch.patch"
done
git -C "$NI_RUA_ROBODOJO_ROOT/third_party/IsaacLab" apply --check "$NI_RUA_REPO_ROOT/docs/development/patches/isaaclab.patch" &&
git -C "$NI_RUA_ROBODOJO_ROOT/third_party/IsaacLab" apply "$NI_RUA_REPO_ROOT/docs/development/patches/isaaclab.patch"
```

若 `apply --check` 失败，通常是因为补丁已经应用过，或者代码检出版本与上表中的固定版本不匹配；请勿强制打补丁。在多 GPU 机器上，请核实 Vulkan 与 `nvidia-smi` 的设备顺序是否与你的硬件配置一致。

## 将 NI-RUA 指向 RoboDojo

后端路径在统一配置的 `backend` 节点中配置，默认值如下：

| 配置字段 | 默认值 | 环境变量 |
| --- | --- | --- |
| `backend.root` | `~/code/robodojo` | `NI_RUA_ROBODOJO_ROOT` |
| `backend.python` | `~/miniforge3/envs/RoboDojo/bin/python` | `NI_RUA_ROBODOJO_PYTHON` |

若你的安装路径不同，可在 Git 忽略的 `configs/ni-rua.local.json` 或自定义配置文件中进行覆盖：

```json
{"version": 1, "backend": {"root": "/opt/robodojo", "python": "/opt/conda/envs/RoboDojo/bin/python"}}
```

运行 `mamba run -n ni-rua python -m ni_rua.apps.gateway --print-config` 即可在不启动机器人、模型或服务器的情况下核验解析出的实际配置。完整的配置优先级规则请参阅[配置指南](configuration.zh-CN.md#使用方法与覆盖优先级)。

## 验证原生环境

以下测试会读取配置的后端路径，在不运行完整任务的情况下验证数值行为与接口：

```sh
mamba run -n ni-rua python -m unittest discover -s tests -p test_native_pose_devices.py
mamba run -n ni-rua python -m unittest discover -s tests -p test_native_garment_runtime.py
```

跨设备检查额外需要 `NI_RUA_NATIVE_TEST_CUDA=1` 以及 `CUDA_VISIBLE_DEVICES` 中有一个空闲 GPU。通过这些单元测试并不等同于端到端任务成功；请通过[运行指南](running.zh-CN.md#launching-a-single-run)中的单次运行任务或 `scripts/robot_validation.py` 测试真实的机器人动作。

## 可选组件

- **外部感知服务**：默认配置启用了 `features.external_perception`，该开关会查询位于 `http://127.0.0.1:8331` 的本地 VGGT 服务。若该服务未运行，对应字段将被标记为 `unavailable`，评测回合仍会正常继续。安装与启动说明请参阅 [`services/vggt_height/README.md`](../../services/vggt_height/README.md)；若不需要该服务，可在配置中将其设置为 `"external_perception": false`。
- **$\pi_{0.5}$ 本地策略**：默认禁用（`features.pi05=false`）。它在专用的 Python 解释器（`policy.python` / `NI_RUA_PI05_PYTHON`）中使用来自 `XPolicyLab` 的 OpenPI，并搭配适配 ARX X5 的权重（`policy.checkpoint` / `NI_RUA_PI05_CHECKPOINT`）。关于权重的来源与校验和，请参阅 [`pi05_arx_manifest.json`](../../ni_rua/runtime/policy/policy_workers/pi05_arx_manifest.json)；关于服务接口，请参阅 [$\pi_{0.5}$ 推理服务](pi05-inference-service.md)。
