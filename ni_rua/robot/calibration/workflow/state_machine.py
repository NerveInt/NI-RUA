"""SO-101 guided calibration workflow: the steps, modes and pass criteria, declared as data.

Pure module (no numpy, cv2, LeRobot or file access) so the worker, the gateway and the tests share one
definition. The worker executes steps in ``calibration_runner.py``; the page ``/calibration`` renders
:func:`describe` (docs/development/so101-integration.md, "Calibration workflow").

Every pass threshold is a deployment ``calibration`` field (units and ranges in ``deployment.FIELDS``);
nothing here carries a tuning number. Operator-only: none of this ever enters model context.

Steps (ids are stable; order is the first-run order):

* ``S0`` 设备自检: read-only device check.
* ``S1`` 关节零位与桌面高度: fingertip touch-off at ``touch_points_xy_m``.
* ``S2`` 腕部相机内参与手眼: robot grid capture with the board on the table.
* ``S3`` 固定相机内参: hand-held board in front of the fixed camera.
* ``S4`` 固定相机外参: static board seen by both cameras, wrist chain gives the board pose.
* ``S5`` 联合求解与交叉校验: joint offsets + hand-eye + table height, then cross-checks.
* ``S6`` 应用与验证: review, atomic apply (with rollback) and verification touches.
"""
from __future__ import annotations

from copy import deepcopy

STEP_IDS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")

# Step status values and their operator labels.
STATUSES = {"not_started": "未开始", "running": "进行中", "passed": "通过", "failed": "失败", "stale": "已过期"}

# Data dependencies: a step consumes outputs of these steps (of this workflow run when they ran in it,
# otherwise of the active calibration bundle). Re-running a step marks every transitive dependent stale.
DEPENDS_ON = {
    "S0": (),
    "S1": (),
    "S2": (),
    "S3": (),
    "S4": ("S2", "S3"),
    "S5": ("S1", "S2", "S4"),
    "S6": ("S5",),
}

# Live readiness checks. ``auto`` checks are measured by the worker; ``operator`` checks are confirmed by
# the operator on the page (the machine cannot see them) and are re-sent with the start request.
READINESS = {
    "torque_on": {"label": "机械臂已上力（力矩开启）", "kind": "auto"},
    "telemetry_fresh": {"label": "关节遥测新鲜", "kind": "auto"},
    "cameras_streaming": {"label": "两个相机都在出图且尺寸正确", "kind": "auto"},
    "fixed_streaming": {"label": "固定相机在出图且尺寸正确", "kind": "auto"},
    "board_both_visible": {"label": "标定板两相机都可见", "kind": "auto"},
    "touches_available": {"label": "有桌面触碰数据（本轮 S1 或当前标定）", "kind": "auto"},
    "joint_inputs_ready": {"label": "S4 已在本轮通过且未过期", "kind": "auto"},
    "wrist_calibration_available": {"label": "有腕部相机标定（本轮 S2 或当前标定）", "kind": "auto"},
    "fixed_calibration_available": {"label": "有可用固定相机内参（本轮 S3 或明确复用当前版本；失败或过期阶段不能回退）", "kind": "auto"},
    "review_ready": {"label": "S5 已在本轮通过且未过期", "kind": "auto"},
    "applied": {"label": "本轮应用版本、已加载标定和实际关节映射一致", "kind": "auto"},
    "touches_physically_confirmed": {"label": "已事后检查各采集点的影像或外部测量：确实接触桌面，接触部位符合几何模型；仅编码器滞后或原始负载不足以确认", "kind": "operator"},
    "table_clear": {"label": "桌面区域已清空（标定板和物体已移走）", "kind": "operator"},
    "workspace_clear": {"label": "机械臂运动范围内无人手和障碍物", "kind": "operator"},
    "board_matches": {"label": "标定板参数与实物一致（方格、标记边长、板厚）", "kind": "operator"},
}

# Fixed comparison targets of ``threshold_measure`` (value known before the step runs).
THRESHOLD_CONSTANTS = {"one": 1}
# Operator labels of ``threshold_measure`` targets that are runner outputs (value only known after the run).
THRESHOLD_MEASURE_LABELS = {"planned_points": "计划点数"}


def _criterion(identifier, label, measure, op, *, threshold=None, threshold_measure=None, unit="", scale=1.0):
    """One pass criterion: ``measure`` (runner output key) compared with a deployment ``calibration``
    field ``threshold`` or another measure ``threshold_measure``. ``op``: ``<=``, ``>=``, ``==``, ``true``
    or ``info`` (shown with the result, never fails the step). ``scale``/``unit`` only affect display (e.g. m shown as mm with scale 1000)."""
    return dict(id=identifier, label=label, measure=measure, op=op, threshold=threshold,
                threshold_measure=threshold_measure, unit=unit, scale=scale)


STEPS = (
    dict(id="S0", key="self_check", title="设备自检", motion=False,
         purpose="确认舵机总线、两个相机和关节遥测都正常，再开始任何标定动作。只读取，机械臂不动。",
         operator=["打开舵机 12 V 电源，确认 USB 线已连接", "确认腕部相机和固定相机都已连接、镜头无遮挡"],
         machine="读取上电自检报告；实时读取舵机电压、温度和力矩状态；从每个相机取一帧新图像检查尺寸；读取一次关节角确认遥测新鲜。",
         readiness=[], outputs=["自检报告"],
         criteria=[_criterion("preflight_ok", "上电自检报告通过", "preflight_ok", "true"),
                   _criterion("power_ok", "舵机电压与温度在部署 preflight 范围内", "power_ok", "true"),
                   _criterion("torque_ok", "力矩已开启", "torque_ok", "true"),
                   _criterion("telemetry_age_s", "关节读数延迟", "telemetry_age_s", "<=",
                              threshold="selfcheck_max_telemetry_age_s", unit="s"),
                   _criterion("cameras_ok", "每个相机出图且尺寸等于部署配置", "cameras_ok", "true")]),
    dict(id="S1", key="touch_off", title="关节零位与桌面高度", motion=True,
         purpose="采集多个位置的阻滞候选及完整轨迹，估算 FK/CAD 高度。采集通过不代表物理接触已确认；S5 前须检查独立支持证据。",
         operator=["移走机械臂前方桌面上的标定板和所有物体", "手离开机械臂，站在急停旁边"],
         machine="依次移动到采集点上方，闭合夹爪，小步下降并记录命令、测量和阻滞证据；阻滞不是桌面接触真值，原始负载不是力。保存轨迹后抬起并回到 home。",
         readiness=["torque_on", "telemetry_fresh", "table_clear", "workspace_clear"],
         outputs=["每个点的接触关节角与接触点", "名义桌面高度"],
         criteria=[_criterion("contacts", "完成阻滞候选采集的点数（待物理审核）", "contacts", "==", threshold_measure="planned_points")]),
    dict(id="S2", key="wrist", title="腕部相机内参与手眼", motion=True,
         purpose="标定腕部相机的内参（焦距、畸变）和它相对夹爪的安装位姿（手眼）。",
         operator=["把标定板平放在桌面上，两个相机都能看到，固定住", "核对下方的标定板参数与实物一致（可修改）",
                   "开始后不要移动标定板"],
         machine="机械臂以降低的速度走一组位姿，每个位姿稳定后两个相机各拍一张；回到 home 后求解腕部内参与手眼。",
         readiness=["torque_on", "board_both_visible", "board_matches", "workspace_clear"],
         outputs=["腕部相机内参", "手眼 T_link_camera", "标定板视图（供 S5）"],
         criteria=[_criterion("kept", "有效样本数", "kept", ">=", threshold="min_samples"),
                   _criterion("wrist_intrinsics_rms_px", "腕部内参 RMS", "wrist_intrinsics_rms_px", "<=",
                              threshold="max_intrinsics_rms_px", unit="px"),
                   _criterion("hand_eye_translation_rms_m", "手眼平移 RMS", "hand_eye_translation_rms_m", "<=",
                              threshold="max_hand_eye_translation_m", unit="mm", scale=1000.0),
                   _criterion("hand_eye_rotation_rms_rad", "手眼旋转 RMS", "hand_eye_rotation_rms_rad", "<=",
                              threshold="max_hand_eye_rotation_rad", unit="rad")]),
    dict(id="S3", key="fixed_intrinsics", title="固定相机内参", motion=False,
         purpose="标定固定相机自己的内参。固定相机看静止的标定板只有一个视角，所以要手持标定板在它前面换多个位置和角度。",
         operator=["拿起标定板放到固定相机前，在画面不同位置停住约 1 秒", "也要靠近、远离，并倾斜标定板",
                   "覆盖足够后点「求解」，然后把标定板平放回桌面（S4 需要）"],
         machine="从固定相机视频流中自动挑选拿稳且不重复的视图，给出覆盖度和提示；求解固定相机内参。机械臂不动。",
         readiness=["fixed_streaming"], outputs=["固定相机内参"],
         criteria=[_criterion("views", "保留视图数", "views", ">=", threshold="handheld_min_views"),
                   _criterion("fixed_intrinsics_rms_px", "固定相机内参 RMS", "fixed_intrinsics_rms_px", "<=",
                              threshold="max_handheld_intrinsics_rms_px", unit="px")]),
    dict(id="S4", key="fixed_extrinsic", title="固定相机外参", motion=True,
         purpose="求固定相机在机械臂坐标系中的位置和朝向：标定板静止在桌面上，腕部相机通过机械臂运动链给出标定板位置，固定相机同时看到它。",
         operator=["把标定板平放在桌面上，两个相机都能看到，固定住", "开始后不要移动标定板"],
         machine="机械臂走几个位姿，每个位姿两个相机同时拍照；用当前（或本轮 S3 的）固定相机内参求解外参并计算链路重投影误差。",
         readiness=["torque_on", "wrist_calibration_available", "fixed_calibration_available", "board_both_visible", "workspace_clear"],
         outputs=["固定相机外参 T_base_camera", "标定板视图（供 S5）"],
         # The chain residual here still contains any joint-zero error; S5 re-solves the extrinsic with the
         # corrected offsets and gates it on max_fixed_reprojection_px. Reported here for information only.
         criteria=[_criterion("fixed_samples", "固定相机看到标定板的样本数", "fixed_samples", ">=", threshold_measure="one"),
                   _criterion("fixed_chain_reprojection_px", "固定相机链路重投影 RMS（参考，S5 按新零位复核门限）",
                              "fixed_chain_reprojection_px", "info", unit="px")]),
    dict(id="S5", key="joint_solve", title="联合求解与交叉校验", motion=False,
         purpose="把标定板视图、桌面触碰点和“板面 = 桌面 + 板厚”先验放进同一个最小二乘问题，同时估计关节零位偏差、手眼和桌面高度，再做交叉校验。",
         operator=["事后检查各采集点的影像或外部测量，确认桌面接触及接触部位，不能只根据滞后或负载打勾", "确认后等待求解完成（机械臂不动）"],
         machine="联合求解后检查：板面高度是否等于桌面加板厚；触碰残差；固定相机链路重投影；关节零位是否在合理范围。任何一项不通过都会说明原因。",
         readiness=["joint_inputs_ready", "touches_available", "touches_physically_confirmed"],
         outputs=["关节零位 offset_rad", "手眼", "桌面高度", "固定相机外参（按新零位）"],
         criteria=[_criterion("joint_reprojection_px", "腕部重投影 RMS", "joint_reprojection_px", "<=",
                              threshold="max_joint_reprojection_px", unit="px"),
                   _criterion("board_height_error_m", "板面高度 − (桌面 + 板厚)", "board_height_error_m", "<=",
                              threshold="max_board_height_error_m", unit="mm", scale=1000.0),
                   _criterion("max_touch_residual_m", "最大触碰残差", "max_touch_residual_m", "<=",
                              threshold="max_touch_residual_m", unit="mm", scale=1000.0),
                   _criterion("fixed_chain_reprojection_px", "固定相机链路重投影 RMS", "fixed_chain_reprojection_px", "<=",
                              threshold="max_fixed_reprojection_px", unit="px"),
                   _criterion("max_joint_offset_rad", "最大关节零位 |offset_rad|", "max_joint_offset_rad", "<=",
                              threshold="max_joint_offset_rad", unit="rad")]),
    dict(id="S6", key="apply_verify", title="应用与验证", motion=True,
         purpose="应用新标定后在未参与拟合的点检查 FK/CAD 高度一致性。仍共享接触检测和几何假设，不是独立物理精度证明；5 mm 等验收界限保持原配置。",
         operator=["查看新旧数值对比，确认后点「应用」", "应用后移走桌面上的标定板和物体，再开始验证触碰",
                   "验证不通过时可一键回滚到上一个版本"],
         machine="写入新版本并切换 active，重载相机标定，按新关节零位更新部署文件；在验证点触碰桌面，比较接触点高度与新桌面高度。",
         readiness=["applied", "torque_on", "table_clear", "workspace_clear"],
         outputs=["已生效的标定版本", "验证触碰结果"],
         criteria=[_criterion("contacts", "接触成功的验证点数", "contacts", "==", threshold_measure="planned_points"),
                   _criterion("max_verify_error_m", "最大验证误差 |接触高度 − 桌面高度|", "max_verify_error_m", "<=",
                              threshold="max_verify_touch_error_m", unit="mm", scale=1000.0)]),
)
STEP_BY_ID = {step["id"]: step for step in STEPS}

MODES = {
    "first_run": {"title": "首次完整标定", "description": "第一次使用或全部重来：按顺序完成 S0–S6。"},
    "recalibrate": {"title": "重新标定（选择哪里变了）", "description": "只重做受影响的步骤。"},
}

# Recalibration: what changed -> steps to run (union over the chosen changes, in STEP_IDS order).
CHANGES = {
    "fixed_moved": {"label": "固定相机被移动", "steps": ("S4", "S5", "S6")},
    "fixed_replaced": {"label": "固定相机换了 / 调了焦", "steps": ("S3", "S4", "S5", "S6")},
    "wrist_bumped": {"label": "腕部相机被碰", "steps": ("S2", "S4", "S5", "S6")},
    "table_or_base_moved": {"label": "换了桌子或底座挪了", "steps": ("S1", "S2", "S4", "S5", "S6")},
    "board_changed": {"label": "换了标定板", "steps": ("S2", "S4", "S5", "S6")},
    "everything": {"label": "全部重来", "steps": STEP_IDS},
}


def plan_for(mode, changes=()):
    """Ordered step ids for ``mode`` (``first_run`` or ``recalibrate`` with ``changes`` from :data:`CHANGES`).

    Example: ``plan_for("recalibrate", ["fixed_moved", "wrist_bumped"])`` -> ``["S2", "S4", "S5", "S6"]``.
    """
    if mode == "first_run":
        return list(STEP_IDS)
    if mode != "recalibrate":
        raise ValueError(f"calibration_mode_unknown:{mode}")
    changes = list(changes or ())
    if not changes:
        raise ValueError("calibration_recalibrate_requires_changes")
    unknown = [change for change in changes if change not in CHANGES]
    if unknown:
        raise ValueError("calibration_change_unknown:" + ",".join(unknown))
    wanted = {step for change in changes for step in CHANGES[change]["steps"]}
    return [step for step in STEP_IDS if step in wanted]


def dependents(step_id):
    """Every step that (transitively) consumes ``step_id``'s outputs, in STEP_IDS order."""
    if step_id not in STEP_BY_ID:
        raise ValueError(f"calibration_step_unknown:{step_id}")
    found, frontier = set(), {step_id}
    while frontier:
        frontier = {step for step, needs in DEPENDS_ON.items() if set(needs) & frontier} - found
        found |= frontier
    return [step for step in STEP_IDS if step in found]


def mark_stale(statuses, step_id):
    """Return a copy of ``statuses`` ({step: status}) where every dependent of ``step_id`` that has a
    result (``passed`` or ``failed``) becomes ``stale``; not-started dependents stay ``not_started``."""
    out = dict(statuses)
    for step in dependents(step_id):
        if out.get(step) in ("passed", "failed"):
            out[step] = "stale"
    return out


def _compare(value, op, limit):
    if op == "info":
        return True
    if op == "true":
        return value is True
    if value is None or limit is None:
        return False
    try:
        value, limit = float(value), float(limit)
    except (TypeError, ValueError):
        return False
    return {"<=": value <= limit, ">=": value >= limit, "==": value == limit}[op]


def evaluate(step_id, measures, settings):
    """Judge one step result. ``measures``: runner outputs (SI units); ``settings``: deployment ``calibration``.

    Returns ``{"passed": bool, "criteria": [{id, label, value, threshold, op, unit, scale, passed}]}``. A
    missing measure fails its criterion (``value`` None), so an incomplete step is never ``passed``.
    """
    rows = []
    constants = THRESHOLD_CONSTANTS
    for criterion in STEP_BY_ID[step_id]["criteria"]:
        value = measures.get(criterion["measure"])
        if criterion["threshold"] is not None:
            limit = settings[criterion["threshold"]]
        elif criterion["threshold_measure"] is not None:
            limit = constants.get(criterion["threshold_measure"], measures.get(criterion["threshold_measure"]))
        else:
            limit = None
        rows.append(dict(id=criterion["id"], label=criterion["label"], value=value, threshold=limit,
                         threshold_key=criterion["threshold"], op=criterion["op"], unit=criterion["unit"],
                         scale=criterion["scale"], passed=_compare(value, criterion["op"], limit)))
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "criteria": rows}


def describe(settings):
    """JSON-able workflow definition for the page: steps (with thresholds filled in), modes, changes,
    readiness labels and status labels. ``settings``: deployment ``calibration`` section."""
    steps = []
    for step in STEPS:
        item = deepcopy(step)
        item["depends_on"] = list(DEPENDS_ON[step["id"]])
        item["readiness"] = [dict(id=check, **READINESS[check]) for check in step["readiness"]]
        for criterion in item["criteria"]:
            target = criterion["threshold_measure"]
            if criterion["threshold"]:
                criterion["threshold_value"] = settings.get(criterion["threshold"])
            else:
                criterion["threshold_value"] = THRESHOLD_CONSTANTS.get(target)
            # Name of a run-time comparison target (e.g. contacts == planned points), shown when there is no value yet.
            criterion["threshold_label"] = THRESHOLD_MEASURE_LABELS.get(target) if target else None
        steps.append(item)
    return {"steps": steps, "modes": deepcopy(MODES),
            "changes": [dict(id=key, label=value["label"], steps=list(value["steps"])) for key, value in CHANGES.items()],
            "statuses": dict(STATUSES)}
