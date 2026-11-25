"""中间态切片与离线感知重算模块（backend-neutral store & perception counterfactual slicing）。

支持从已归档实验目录的任意工具调用节点（call_id）或动作节点（action_id）：
1. 裁剪 SQLite 事件库并提取前置物理控制序列（prefix_commands），支持无相机渲染极速物理快进；
2. 用当前工作区新代码重算切点处的工具输出（如 object_table / stereo_matches / obs3d 随路感知）；
3. 通过 ``codex_fork`` 裁剪 Codex rollout JSONL 轨迹并通过 ``thread/fork`` 分叉新会话继续推理。
"""


from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from ...core.presentation.actor_output import ActorOutput
from ...core.presentation.actor_presentation import ActorPresentation
from ...core.config import Config
from ...core.contracts import canonical, digest
from ...perception.scene.store import SceneStore
from ...core.storage.store import Store, _file_identity, open_read_store
from ...core.tool_output import wire_content


class UnsupportedSliceToolError(ValueError):
    """Raised when ``recompute_tool_output`` does not support recomputing a tool kind."""


def list_run_slices(source_run: str | Path) -> dict:
    """列出归档运行中所有可切片的工具调用、观察与动作节点。"""
    source_run = Path(source_run).resolve()
    with open_read_store(source_run) as store:
        identity = store.read_state("runtime", "identity") or {}
        store.identity = identity
        observations = {rec["observation_id"]: rec for rec in store.records("observation")}
        calls = []
        for idx, rec in enumerate(store.records("tool_call")):
            params = rec.get("params") or {}
            cid = params.get("callId")
            tool = params.get("tool")
            args = params.get("arguments") or {}
            req = args.get("request") if isinstance(args.get("request"), dict) else args
            calls.append({
                "index": idx,
                "call_id": cid,
                "turn_id": params.get("turnId"),
                "tool": tool,
                "kind": req.get("kind") if isinstance(req, dict) else None,
                "mode": req.get("mode") if isinstance(req, dict) else None,
                "observation_id": req.get("observation_id") if isinstance(req, dict) else None,
                "labels": req.get("labels") if isinstance(req, dict) else None,
            })
        actions = []
        for idx, req in enumerate(store.records("action_request")):
            aid = req["action_id"]
            try:
                rcpt = store.read("action_receipt", aid)
            except KeyError:
                rcpt = {}
            res_obs_id = rcpt.get("result_observation_id")
            res_obs = observations.get(res_obs_id) or {}
            sim_step = ((res_obs.get("robot_state") or {}).get("sim_step"))
            actions.append({
                "index": idx,
                "action_id": aid,
                "based_on_observation_id": req.get("based_on_observation_id"),
                "status": rcpt.get("status"),
                "result_observation_id": res_obs_id,
                "sim_step": sim_step,
                "executed_native_steps": rcpt.get("executed_native_steps"),
            })
        return {
            "run_dir": str(source_run),
            "identity": identity,
            "tool_calls": calls,
            "actions": actions,
            "observation_count": len(observations),
        }


def extract_prefix_commands(store: Store, until_action_id: str | None = None) -> dict:
    """提取截止到指定动作（含）的所有底层控制步序列（dispatched_trace）及目标边界状态。"""
    commands = []
    last_action_id = None
    last_receipt = None
    for req in store.records("action_request"):
        aid = req["action_id"]
        try:
            rcpt = store.read("action_receipt", aid)
        except KeyError:
            break
        trace_ref = rcpt.get("dispatched_trace_ref")
        if trace_ref:
            _, raw_bytes = store.resolve_artifact(trace_ref, ("actor", "internal", "observer"))
            trace = json.loads(raw_bytes)
            if isinstance(trace, list):
                commands.extend(trace)
            elif isinstance(trace, dict) and isinstance(trace.get("samples"), list):
                commands.extend(trace["samples"])
        last_action_id = aid
        last_receipt = rcpt
        if until_action_id and aid == until_action_id:
            break
    if until_action_id and last_action_id != until_action_id:
        raise KeyError(f"未找到目标动作: {until_action_id}")

    observations = store.records("observation")
    boundary_obs = observations[0] if observations else None
    if last_receipt and last_receipt.get("result_observation_id"):
        target_obs_id = last_receipt["result_observation_id"]
        for obs in observations:
            if obs["observation_id"] == target_obs_id:
                boundary_obs = obs
                break
    elif last_action_id:
        for obs in observations:
            if obs.get("last_action_id") == last_action_id:
                boundary_obs = obs

    robot_state = (boundary_obs or {}).get("robot_state") or {}
    if not robot_state and boundary_obs and boundary_obs.get("robot_state_ref"):
        try:
            _, raw_state = store.resolve_artifact(boundary_obs["robot_state_ref"], ("actor", "internal", "observer"))
            robot_state = json.loads(raw_state)
        except Exception:
            robot_state = {}
    arms = robot_state.get("arms") or {}
    arm_joints = robot_state.get("arm_joint_positions") or {
        side: arm.get("q_rad") for side, arm in arms.items() if isinstance(arm, dict) and arm.get("q_rad") is not None
    }
    link6_poses = robot_state.get("link6_poses") or {
        side: arm.get("ee_pose") for side, arm in arms.items() if isinstance(arm, dict) and arm.get("ee_pose") is not None
    }
    gripper_opening = robot_state.get("gripper_opening_m") or {
        side: (arm.get("gripper") or {}).get("position_m")
        for side, arm in arms.items()
        if isinstance(arm, dict) and isinstance(arm.get("gripper"), dict)
    }
    gripper_goals = {
        side: float((arm.get("gripper") or {}).get("goal_normalized", 1.0))
        for side, arm in arms.items()
        if isinstance(arm, dict) and isinstance(arm.get("gripper"), dict)
    }
    sim_step = (boundary_obs or {}).get("sim_step")
    if sim_step is None:
        sim_step = robot_state.get("sim_step")
    return {
        "until_action_id": last_action_id,
        "command_count": len(commands),
        "target_sim_step": commands[-1]["sim_step"] if commands else int(sim_step or 0),
        "boundary_observation_id": (boundary_obs or {}).get("observation_id"),
        "target_robot_state": {
            "sim_step": sim_step,
            "arm_joint_positions": arm_joints,
            "link6_poses": link6_poses,
            "gripper_opening_m": gripper_opening,
            "gripper_goals": gripper_goals,
        },
        "commands": commands,
    }


def _hydrate_runtime_perception_state(runtime, store: Store, presenter: ActorOutput, before_call_id: str | None):
    """恢复切点之前的 object_table 观察缓存与动作状态账本（不做跨帧标签隐式注入）。"""
    from ...runtime.actions.action_state_ledger import (
        compute_object_table_delta,
        record_execution_outcome,
        record_object_table_delta,
    )
    from ...perception.external.obs3d import _remember, attached_table_view

    obs_tables = {}
    prefix_action_ids = set()
    for rec in store.records("tool_result"):
        if before_call_id is not None and (rec.get("correlation") or {}).get("tool_call_id") == before_call_id:
            break
        if not rec.get("raw_ref"):
            continue
        try:
            raw = json.loads(store.resolve_artifact(rec["raw_ref"], ("observer", "actor"))[1])
            data = presenter.restore(raw.get("data") or {})
            if isinstance(data, dict):
                act_id = (
                    data.get("action_id")
                    or (data.get("receipt") or {}).get("action_id")
                    or (data.get("action_receipt") or {}).get("action_id")
                )
                if act_id:
                    prefix_action_ids.add(act_id)
                for ra in (data.get("action_summary") or {}).get("recent_actions") or []:
                    if isinstance(ra, dict) and ra.get("action_id"):
                        prefix_action_ids.add(ra["action_id"])
            if isinstance(data, dict) and data.get("query_type") in ("object_table", "reanchor_tracks") and isinstance(data.get("value"), dict):
                val = data["value"]
                obs_ids = data.get("observation_ids") or []
                if obs_ids and val.get("rows") is not None:
                    shown = attached_table_view(val)
                    obs_tables[obs_ids[0]] = shown
                    _remember(runtime, obs_ids[0], "object_table", (shown, None, data))
            else:
                for item in (raw.get("wire_response") or {}).get("contentItems") or []:
                    if item.get("type") == "inputText" and '"object_table"' in str(item.get("text", "")):
                        parsed = presenter.restore(json.loads(item["text"]))
                        ot = parsed.get("object_table") if isinstance(parsed, dict) else None
                        obs_id = parsed.get("observation_id") if isinstance(parsed, dict) else None
                        if isinstance(ot, dict) and obs_id and ot.get("rows") is not None:
                            obs_tables[obs_id] = ot
                            _remember(runtime, obs_id, "object_table", (ot, None, {}))
        except Exception:
            pass

    if not hasattr(runtime, "unified_action_plans") or not isinstance(getattr(runtime, "unified_action_plans", None), dict):
        runtime.unified_action_plans = {}
    kept_obs_ids = {o.get("observation_id") for o in store.records("observation") if isinstance(o, dict) and o.get("observation_id")}
    for plan_rec in store.records("unified_action_plan"):
        if kept_obs_ids and plan_rec.get("observation_id") not in kept_obs_ids:
            continue
        pid = (plan_rec.get("summary") or {}).get("plan_id")
        if pid:
            runtime.unified_action_plans[pid] = deepcopy(plan_rec)
            robot = getattr(runtime, "robot", None)
            if robot is not None and isinstance(getattr(robot, "_batch_plans", None), dict) and plan_rec.get("native_plan_id"):
                robot._batch_plans.setdefault(
                    plan_rec["native_plan_id"],
                    {
                        "plan_id": plan_rec["native_plan_id"],
                        "status": "planned",
                        "source_action": deepcopy(plan_rec.get("action") or {}),
                        "source_stages": deepcopy((plan_rec.get("action") or {}).get("stages") or []),
                        "completed_stage_ids": deepcopy(plan_rec.get("completed_stage_ids") or []),
                    },
                )

    for idx, req in enumerate(store.records("action_request")):
        aid = req.get("action_id")
        if not aid:
            continue
        try:
            rcpt = store.read("action_receipt", aid)
        except KeyError:
            continue
        a_obs = rcpt.get("result_observation_id")
        if before_call_id is not None and aid not in prefix_action_ids and a_obs not in obs_tables:
            continue
        record_execution_outcome(runtime, f"hydrated-plan-{idx}", req, rcpt)
        b_obs = req.get("based_on_observation_id")
        if b_obs in obs_tables and a_obs in obs_tables and b_obs != a_obs:
            delta = compute_object_table_delta(obs_tables[b_obs], obs_tables[a_obs], before_observation_id=b_obs)
            if delta:
                record_object_table_delta(runtime, b_obs, a_obs, delta)


def archived_embodiment(store, run_dir):
    """The ``EmbodimentProfile`` the archived run used, restored from its ``embodiment_profile-<digest>`` configuration
    record (``manifest.json`` ``embodiment_record_id``, else the store's last such record;
    ``embodiments.profile_from_record``), never rebuilt from the current code or deployment files. None for an archive
    written before the profile record existed: offline recomputation then has no profile and perception keeps the
    rules that predate it (``object_table.default_camera``: an ``so101`` observation selects ``fixed``; profile
    values otherwise default to the ARX-X5 robot, ``perception.external.embodiment_of``)."""
    from ...robot.model.embodiments import archived_embodiment_record, profile_from_record
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    _, record = archived_embodiment_record(store, manifest.get("embodiment_record_id"))
    return profile_from_record(record) if record is not None else None


def recompute_tool_output(source_run: str | Path, call_id: str, *, config_overrides: dict | None = None) -> dict:
    """在已归档运行的数据上，使用当前工作区代码重新计算指定 tool_call 的输出。"""
    source_run = Path(source_run).resolve()
    with open_read_store(source_run) as store:
        identity = store.read_state("runtime", "identity") or {}
        store.identity = identity
        cfg_dict = json.loads((source_run / "effective_config.json").read_text())
        cfg = Config.from_dict(cfg_dict)
        if config_overrides:
            cfg = cfg.with_overrides(config_overrides)
        # 使用内存态叠加层包装只读 Store，支持 ActorPresentation 分配新短引用而不写盘
        mem_states = {}

        def _read_state(namespace, key, default=None):
            if (namespace, key) in mem_states:
                return mem_states[(namespace, key)]
            return store.read_state(namespace, key, default)

        def _set_state(namespace, key, value):
            mem_states[(namespace, key)] = value

        def _states(namespace):
            base = {item.get("alias") or item.get("ref"): item for item in store.states(namespace)}
            for (ns, k), val in mem_states.items():
                if ns == namespace:
                    base[k] = val
            return list(base.values())

        def _put_artifact_noop(data, kind, mime_type, scope="actor"):
            return f"artifact-recomputed-{digest({'kind': kind, 'size': len(data)})[:16]}"

        scene = SceneStore(source_run / "scene", store.resolve_artifact, _put_artifact_noop)
        mem_store = SimpleNamespace(
            root=source_run,
            identity=identity,
            lock=store.lock,
            resolve_artifact=store.resolve_artifact,
            resolve_json=scene.resolve_json,
            put_artifact=_put_artifact_noop,
            event=lambda *a, **k: None,
            read_state=_read_state,
            set_state=_set_state,
            states=_states,
        )
        pres = ActorPresentation(mem_store)
        out = ActorOutput(mem_store, pres, cfg.features.actor_payload_profile)
        runtime = SimpleNamespace(
            config=cfg,
            identity=identity,
            scene=scene,
            store=mem_store,
            image_settings=cfg.images.to_dict(),
        )
        # The archived embodiment (camera, frame, gripper and hint facts of perception); absent in older archives.
        profile = archived_embodiment(store, source_run)
        if profile is not None:
            runtime.embodiment = profile
        _hydrate_runtime_perception_state(runtime, store, out, call_id)

        call_rec = store.read("tool_call", call_id)
        params = call_rec["params"]
        tool_name = params["tool"]
        restored_args = out.restore(params.get("arguments") or {})

        # 读取旧归档输出以便对比
        old_raw = None
        for rec in store.records("tool_result"):
            if (rec.get("correlation") or {}).get("tool_call_id") == call_id and rec.get("raw_ref"):
                old_raw = json.loads(store.resolve_artifact(rec["raw_ref"], ("observer", "actor"))[1])
                break

        if tool_name == "geometry_query":
            req = restored_args.get("request") if isinstance(restored_args.get("request"), dict) else restored_args
            if req.get("kind") == "object_table":
                from ...perception.external.object_table import object_table_query
                from ...core.storage.images import image_url
                from ...core.tool_output import StructuredText, ToolResponse
                res, overlays = object_table_query(runtime, req)
                content = [StructuredText(res, projection="geometry_result")]
                for cam_id, encoded, _ in overlays:
                    url, _ = image_url(encoded, runtime.image_settings)
                    content.append({"type": "inputImage", "imageUrl": url})
                presented = pres.present(out.present(ToolResponse(content, data=res), tool=tool_name))
                return {
                    "call_id": call_id,
                    "tool": tool_name,
                    "kind": "object_table",
                    "request": req,
                    "new_data": res,
                    "old_data": (old_raw or {}).get("data"),
                    "wire_output": presented,
                    "active_labels": getattr(runtime, "_active_object_labels", None),
                }
        elif tool_name in {"robot_observe", "robot_action", "program_run"}:
            # 支持对观察、动作或程序调用末帧重新计算 obs3d 随路感知与动作状态账本
            from ...perception.external import obs3d
            from ...runtime.actions.action_state_ledger import attach_execution_ledger, record_execution_outcome
            obs_id = None
            cut_req = None
            cut_rcpt = None
            cut_plan_id = "recomputed-plan"
            if tool_name == "robot_observe":
                obs_id = restored_args.get("observation_id")
            if old_raw:
                data = out.restore(old_raw.get("data") or {})
                if isinstance(data, dict):
                    if not obs_id:
                        obs_id = (
                            data.get("observation_id")
                            or data.get("final_observation_id")
                            or (data.get("observation") or {}).get("observation_id")
                            or (data.get("final_observation") or {}).get("observation_id")
                            or (data.get("action_receipt") or {}).get("result_observation_id")
                            or (data.get("receipt") or {}).get("result_observation_id")
                        )
                    if tool_name in {"robot_action", "program_run"}:
                        act_id = (
                            data.get("action_id")
                            or (data.get("receipt") or {}).get("action_id")
                            or (data.get("action_receipt") or {}).get("action_id")
                        )
                        if not act_id:
                            recent_acts = (data.get("action_summary") or {}).get("recent_actions") or []
                            if recent_acts and isinstance(recent_acts[-1], dict):
                                act_id = recent_acts[-1].get("action_id")
                        cut_plan_id = str(data.get("plan_id") or "recomputed-plan")
                        cut_req = data.get("request") if isinstance(data.get("request"), dict) else None
                        cut_rcpt = (
                            data.get("receipt") if isinstance(data.get("receipt"), dict)
                            else data.get("action_receipt") if isinstance(data.get("action_receipt"), dict)
                            else None
                        )
                        if act_id and cut_req is None:
                            try:
                                cut_req = store.read("action_request", act_id)
                            except KeyError:
                                pass
                        if act_id and cut_rcpt is None:
                            try:
                                cut_rcpt = store.read("action_receipt", act_id)
                            except KeyError:
                                pass
                if not obs_id:
                    for item in (old_raw.get("wire_response") or {}).get("contentItems") or []:
                        if item.get("type") == "inputText" and '"observation_id"' in str(item.get("text", "")):
                            try:
                                parsed_txt = out.restore(json.loads(item["text"]))
                                if isinstance(parsed_txt, dict):
                                    obs_id = (
                                        parsed_txt.get("observation_id")
                                        or (parsed_txt.get("observation") or {}).get("observation_id")
                                        or (parsed_txt.get("action_receipt") or {}).get("result_observation_id")
                                    )
                                    if obs_id:
                                        break
                            except Exception:
                                pass
            if not obs_id:
                observations = store.records("observation")
                obs_id = observations[-1]["observation_id"] if observations else None
            bundle = next((o for o in store.records("observation") if o["observation_id"] == obs_id), None)
            if bundle is None:
                raise KeyError(f"未找到观察记录: {obs_id}")
            before_obs_id = cut_req.get("based_on_observation_id") if isinstance(cut_req, dict) else None
            if isinstance(cut_req, dict) and isinstance(cut_rcpt, dict):
                record_execution_outcome(runtime, cut_plan_id, cut_req, cut_rcpt)
            parts = obs3d.attachments(
                runtime,
                bundle,
                {"tool_call_id": call_id},
                reply="observe" if tool_name == "robot_observe" else "action",
                before_observation_id=before_obs_id,
            )
            wire_parts = pres.present(wire_content(parts)) if parts else []
            new_obs3d = deepcopy(parts[0].value) if parts and hasattr(parts[0], "value") else None
            ledger_attachment = {}
            if isinstance(cut_req, dict) and isinstance(cut_rcpt, dict):
                ledger_attachment = attach_execution_ledger(runtime, {"receipt": deepcopy(cut_rcpt)})
                if isinstance(new_obs3d, dict):
                    if "repeat_failure_warning" in ledger_attachment:
                        new_obs3d["repeat_failure_warning"] = ledger_attachment["repeat_failure_warning"]
                    if "episode_ledger" in ledger_attachment:
                        new_obs3d["episode_ledger"] = ledger_attachment["episode_ledger"]
            wire_output = deepcopy((old_raw or {}).get("wire_response"))
            if isinstance(wire_output, dict) and isinstance(wire_output.get("contentItems"), list) and wire_parts:
                # 替换历史工具输出中的 obs3d JSON 文本段，并将最新 repeat_failure_warning/episode_ledger 注入动作回执段
                items = pres.present(wire_output["contentItems"])
                if ledger_attachment and items and items[0].get("type") == "inputText":
                    try:
                        first_obj = json.loads(items[0].get("text", ""))
                        if isinstance(first_obj, dict) and ("action_id" in first_obj or "receipt" in first_obj):
                            if "repeat_failure_warning" in ledger_attachment:
                                first_obj["repeat_failure_warning"] = pres.present(ledger_attachment["repeat_failure_warning"])
                            rcpt_view = first_obj.get("receipt") or {}
                            if (
                                rcpt_view.get("batch_status", rcpt_view.get("status")) in ("stopped", "failed")
                                or rcpt_view.get("checkpoint_reason") in ("aperture_below_min_m", "aperture_above_max_m")
                            ) and "episode_ledger" in ledger_attachment:
                                first_obj["episode_ledger"] = pres.present(ledger_attachment["episode_ledger"])
                            items[0] = {"type": "inputText", "text": canonical(first_obj)}
                    except Exception:
                        pass
                replaced = False
                for idx, item in enumerate(items):
                    if item.get("type") == "inputText" and '"object_table"' in str(item.get("text", "")):
                        items[idx] = wire_parts[0]
                        replaced = True
                        break
                if not replaced:
                    items.extend(wire_parts)
                wire_output = {"success": True, "contentItems": items}
            return {
                "call_id": call_id,
                "tool": tool_name,
                "kind": "obs3d_attachments",
                "observation_id": obs_id,
                "new_data": new_obs3d,
                "old_data": (old_raw or {}).get("data"),
                "wire_parts": wire_parts,
                "wire_output": wire_output,
                "active_labels": getattr(runtime, "_active_object_labels", None),
            }
        raise UnsupportedSliceToolError(f"暂不支持重算该工具类型: {tool_name}")


_TRANSIENT_RUNTIME_STATE_SQL = (
    "DELETE FROM state WHERE kind = 'runtime' AND id IN ('app_server', 'robot', 'model_requests', 'scene_obstacles', 'agent')"
)


def _truncate_sqlite_store(db_path: Path, until_call_id: str | None = None, until_action_id: str | None = None) -> dict:
    """在克隆的 events.sqlite 上精确截断切点之后的事件与记录。"""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cut_seq = None
        resolved_call_id = until_call_id
        rows = cur.execute("SELECT seq, body FROM events ORDER BY seq").fetchall()
        if until_call_id:
            for seq, body_s in rows:
                ev = json.loads(body_s) if body_s else {}
                corr = ev.get("correlation") or {}
                payload = ev.get("payload") or {}
                if corr.get("tool_call_id") == until_call_id:
                    cut_seq = seq
                elif ev.get("type") == "codex.raw" and payload.get("direction") == "sent":
                    msg = payload.get("message") or {}
                    if cut_seq is not None and "result" in msg and "id" in msg:
                        cut_seq = seq
                        break
        elif until_action_id:
            last_actor_call_id = None
            target_tool_call_id = None
            for seq, body_s in rows:
                ev = json.loads(body_s) if body_s else {}
                corr = ev.get("correlation") or {}
                payload = ev.get("payload") or {}
                tcid_any = corr.get("tool_call_id") or payload.get("callId")
                if isinstance(tcid_any, str) and tcid_any.startswith("call_"):
                    last_actor_call_id = tcid_any
                elif ev.get("type") == "codex.raw" and payload.get("direction") == "received":
                    msg = payload.get("message") or {}
                    if msg.get("method") == "item/tool/call":
                        cid = (msg.get("params") or {}).get("callId")
                        if isinstance(cid, str) and cid.startswith("call_"):
                            last_actor_call_id = cid
                if corr.get("action_id") == until_action_id:
                    cut_seq = seq
                    tcid = corr.get("tool_call_id")
                    if tcid:
                        target_tool_call_id = tcid
                        resolved_call_id = last_actor_call_id if not str(tcid).startswith("call_") and last_actor_call_id else tcid
                elif target_tool_call_id and corr.get("tool_call_id") in {target_tool_call_id, resolved_call_id}:
                    cut_seq = seq
                elif resolved_call_id and ev.get("type") == "codex.raw" and payload.get("direction") == "sent":
                    msg = payload.get("message") or {}
                    if cut_seq is not None and "result" in msg and "id" in msg:
                        cut_seq = seq
                        break

        if cut_seq is None:
            if until_call_id is None and until_action_id is None:
                cur.execute(_TRANSIENT_RUNTIME_STATE_SQL)
                conn.commit()
            return {"truncated": False, "resolved_call_id": resolved_call_id}

        kept_obs, kept_actions, kept_calls, kept_plans = set(), set(), set(), set()
        last_task_plan = None
        for seq, body_s in rows:
            if seq > cut_seq:
                break
            ev = json.loads(body_s) if body_s else {}
            ev_type = ev.get("type")
            corr = ev.get("correlation") or {}
            payload = ev.get("payload") or {}
            if ev_type in {"observation.available", "observation.published"}:
                oid = corr.get("observation_id") or payload.get("observation_id")
                if oid:
                    kept_obs.add(oid)
            if ev_type in {"action.requested", "action.accepted"} and corr.get("action_id"):
                kept_actions.add(corr["action_id"])
            if ev_type == "task_plan.submitted" and isinstance(payload, dict) and payload.get("plan_id"):
                kept_plans.add(payload["plan_id"])
                last_task_plan = payload
            cid = corr.get("tool_call_id") or payload.get("callId")
            if cid:
                kept_calls.add(cid)

        later_events = cur.execute("SELECT body FROM events WHERE seq > ?", (cut_seq,)).fetchall()
        drop_obs, drop_actions, drop_calls, drop_plans = set(), set(), set(), set()
        for (body_s,) in later_events:
            ev = json.loads(body_s) if body_s else {}
            ev_type = ev.get("type")
            corr = ev.get("correlation") or {}
            payload = ev.get("payload") or {}
            if ev_type in {"observation.available", "observation.published"}:
                oid = corr.get("observation_id") or payload.get("observation_id")
                if oid and oid not in kept_obs:
                    drop_obs.add(oid)
            if ev_type in {"action.requested", "action.accepted"} and corr.get("action_id"):
                if corr["action_id"] not in kept_actions:
                    drop_actions.add(corr["action_id"])
            if ev_type == "task_plan.submitted" and isinstance(payload, dict) and payload.get("plan_id"):
                if payload["plan_id"] not in kept_plans:
                    drop_plans.add(payload["plan_id"])
            cid = corr.get("tool_call_id") or payload.get("callId")
            if cid and cid not in kept_calls:
                drop_calls.add(cid)
            elif ev_type == "codex.raw" and payload.get("direction") == "received":
                msg = payload.get("message") or {}
                if msg.get("method") == "item/tool/call":
                    cid = (msg.get("params") or {}).get("callId")
                    if cid and cid not in kept_calls:
                        drop_calls.add(cid)

        cur.execute("DELETE FROM events WHERE seq > ?", (cut_seq,))
        # runtime/agent is the parent's agent identity (thread, instance, pid); the fork's own runtime/codex,
        # written by ForkedCodexSession.start_forked, is the identity read_agent_identity reports for the slice.
        cur.execute(_TRANSIENT_RUNTIME_STATE_SQL)
        if last_task_plan is not None:
            cur.execute(
                "INSERT OR REPLACE INTO state (kind, id, body) VALUES ('runtime', 'task_plan', ?)",
                (canonical(last_task_plan),),
            )
        else:
            cur.execute("DELETE FROM state WHERE kind = 'runtime' AND id = 'task_plan'")
        for pid in drop_plans:
            cur.execute("DELETE FROM records WHERE kind = 'task_plan' AND id = ?", (pid,))
            cur.execute("DELETE FROM state WHERE kind = 'actor_ref' AND json_extract(body, '$.ref') = ?", (pid,))
        for oid in drop_obs:
            cur.execute("DELETE FROM records WHERE kind = 'observation' AND id = ?", (oid,))
            cur.execute("DELETE FROM records WHERE kind = 'unified_action_plan' AND json_extract(body, '$.observation_id') = ?", (oid,))
            cur.execute("DELETE FROM state WHERE kind = 'actor_ref' AND json_extract(body, '$.ref') = ?", (oid,))
        if not kept_actions:
            cur.execute("DELETE FROM records WHERE kind = 'unified_action_plan'")
            cur.execute("DELETE FROM state WHERE kind = 'runtime' AND id = 'unified_action_sequence'")
        for aid in drop_actions:
            cur.execute("DELETE FROM records WHERE kind IN ('action_request', 'action_receipt') AND id = ?", (aid,))
            cur.execute("DELETE FROM state WHERE kind = 'actor_ref' AND json_extract(body, '$.ref') = ?", (aid,))
        for cid in drop_calls:
            cur.execute("DELETE FROM records WHERE kind IN ('tool_call', 'tool_result') AND id = ?", (cid,))
        conn.commit()
        return {
            "truncated": True,
            "cut_seq": cut_seq,
            "resolved_call_id": resolved_call_id,
            "dropped_observations": len(drop_obs),
            "dropped_actions": len(drop_actions),
            "dropped_tool_calls": len(drop_calls),
            "dropped_task_plans": len(drop_plans),
        }
    finally:
        conn.close()


def prepare_store_slice(
    source_run: str | Path,
    target_run: str | Path,
    *,
    until_call_id: str | None = None,
    until_action_id: str | None = None,
    recompute_cutpoint: bool = False,
    config_overrides: dict | None = None,
) -> dict:
    """从归档实验目录构建后端无关的切片目录（SQLite 裁剪、硬链接资产、前缀控制步与基础 checkpoint_spec.json）。"""
    source_run = Path(source_run).resolve()
    target_run = Path(target_run).resolve()
    created_target = not target_run.exists()
    if not created_target and any(target_run.iterdir()):
        raise ValueError(f"目标切片目录非空: {target_run}")
    target_run.mkdir(parents=True, exist_ok=True)

    # 1. 复制并裁剪 SQLite 数据库（含可能存在的 WAL 文件并校验快照前后一致性），同时若仅指定 until_action_id 则自动解析对应 call_id
    src_db = source_run / "events.sqlite"
    src_wal = source_run / "events.sqlite-wal"
    for db_p in (src_db, src_wal):
        if db_p.is_symlink():
            raise ValueError("database_symlink_denied")
    try:
        src_files = (src_db, src_wal) if src_wal.exists() else (src_db,)
        before_ident = {p: _file_identity(p) for p in src_files}
        shutil.copy2(src_db, target_run / "events.sqlite")
        if src_wal in before_ident:
            shutil.copy2(src_wal, target_run / "events.sqlite-wal")
        after_ident = {p: _file_identity(p) for p in src_files}
        if before_ident != after_ident:
            raise ValueError("archived_database_changed_during_snapshot")
    except FileNotFoundError:
        if not src_wal.exists() and src_db.exists():
            (target_run / "events.sqlite-wal").unlink(missing_ok=True)
            before_db = _file_identity(src_db)
            shutil.copy2(src_db, target_run / "events.sqlite")
            if _file_identity(src_db) != before_db:
                raise ValueError("archived_database_changed_during_snapshot") from None
        else:
            raise
    except Exception:
        for partial in ("events.sqlite", "events.sqlite-wal", "events.sqlite-shm"):
            (target_run / partial).unlink(missing_ok=True)
        if created_target:
            try:
                target_run.rmdir()
            except OSError:
                pass
        raise
    sqlite_info = _truncate_sqlite_store(target_run / "events.sqlite", until_call_id=until_call_id, until_action_id=until_action_id)
    missing_cutpoint = (until_call_id is not None or until_action_id is not None) and not sqlite_info["truncated"]
    if not missing_cutpoint and until_action_id is not None:
        with sqlite3.connect(str(target_run / "events.sqlite")) as check_conn:
            has_receipt = check_conn.execute(
                "SELECT 1 FROM records WHERE kind = 'action_request' AND id = ? "
                "AND EXISTS (SELECT 1 FROM records WHERE kind = 'action_receipt' AND id = ?)",
                (until_action_id, until_action_id),
            ).fetchone()
        if not has_receipt:
            missing_cutpoint = True
    if missing_cutpoint:
        for partial in ("events.sqlite", "events.sqlite-wal", "events.sqlite-shm"):
            (target_run / partial).unlink(missing_ok=True)
        if created_target:
            try:
                target_run.rmdir()
            except OSError:
                pass
        missing = (
            f"until_call_id={until_call_id!r}"
            if until_call_id is not None and not sqlite_info["truncated"]
            else f"until_action_id={until_action_id!r}"
        )
        raise KeyError(f"在 events.sqlite 中未找到目标切点: {missing}")
    effective_call_id = until_call_id or sqlite_info.get("resolved_call_id")

    recomputed = None
    if recompute_cutpoint and effective_call_id:
        try:
            recomputed = recompute_tool_output(source_run, effective_call_id, config_overrides=config_overrides)
        except UnsupportedSliceToolError:
            recomputed = None

    # 2. 硬链接 blobs 与 scene 目录，复制 actor 工作区与配置文件
    for dirname in ("blobs", "scene"):
        src_d = source_run / dirname
        if src_d.exists():
            shutil.copytree(src_d, target_run / dirname, copy_function=os.link)
    if (source_run / "actor").exists():
        shutil.copytree(source_run / "actor", target_run / "actor")
    for fname in ("manifest.json", "effective_config.json"):
        if (source_run / fname).exists():
            shutil.copy2(source_run / fname, target_run / fname)

    # 3. 提取前置物理控制序列与边界状态
    with open_read_store(target_run) as slice_store:
        identity = slice_store.read_state("runtime", "identity") or {}
        slice_store.identity = identity
        prefix_info = extract_prefix_commands(slice_store, until_action_id=until_action_id)

    prefix_path = target_run / "prefix_commands.jsonl"
    prefix_path.write_text(
        "\n".join(json.dumps(row) for row in prefix_info["commands"]) + ("\n" if prefix_info["commands"] else "")
    )

    snap_key = digest({
        "source_run": str(source_run),
        "until_action_id": prefix_info["until_action_id"],
        "target_sim_step": prefix_info["target_sim_step"],
    })[:20]
    shared_snap = Path("/tmp/ni-rua-mujoco-snapshots") / f"{source_run.name}-{prefix_info['until_action_id'] or 'init'}-{snap_key}.npz"
    spec = {
        "schema_version": "0.1",
        "source_run": str(source_run),
        "target_run": str(target_run),
        "until_call_id": effective_call_id,
        "until_action_id": prefix_info["until_action_id"],
        "recompute_cutpoint": recompute_cutpoint,
        "sqlite": sqlite_info,
        "prefix_commands_path": str(prefix_path),
        "mujoco_snapshot_path": str(shared_snap),
        "prefix_command_count": prefix_info["command_count"],
        "target_sim_step": prefix_info["target_sim_step"],
        "boundary_observation_id": prefix_info["boundary_observation_id"],
        "target_robot_state": prefix_info["target_robot_state"],
    }
    (target_run / "checkpoint_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    return {"spec": spec, "recomputed": recomputed}


def prepare_checkpoint_slice(
    source_run: str | Path,
    target_run: str | Path,
    *,
    until_call_id: str | None = None,
    until_action_id: str | None = None,
    recompute_cutpoint: bool = False,
    config_overrides: dict | None = None,
) -> dict:
    """从归档实验目录构建完整的 Codex 切片运行目录（含 SQLite 裁剪、硬链接资产、前缀控制步与截断后的 Codex rollout）。"""
    from .codex_fork import prepare_codex_rollout_slice, require_codex_archive

    source_run = Path(source_run).resolve()
    target_run = Path(target_run).resolve()
    require_codex_archive(source_run)

    prepared = prepare_store_slice(
        source_run,
        target_run,
        until_call_id=until_call_id,
        until_action_id=until_action_id,
        recompute_cutpoint=recompute_cutpoint,
        config_overrides=config_overrides,
    )
    spec = prepared["spec"]
    recomputed = prepared["recomputed"]
    replacement_output = recomputed.get("wire_output") if recomputed else None

    rollout_info = prepare_codex_rollout_slice(
        source_run,
        target_run,
        until_call_id=spec["until_call_id"],
        replacement_output=replacement_output,
    )
    spec["source_thread_id"] = rollout_info["source_thread_id"]
    spec["rollout"] = rollout_info
    (target_run / "checkpoint_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    return {"spec": spec, "recomputed": recomputed}


_CODEX_COMPAT_EXPORTS = frozenset({
    "ForkedCodexSession",
    "_to_rollout_output",
    "find_rollout_path",
    "prepare_codex_rollout_slice",
    "require_codex_archive",
    "slice_rollout_jsonl",
})


def __getattr__(name: str):
    if name in _CODEX_COMPAT_EXPORTS:
        from . import codex_fork
        return getattr(codex_fork, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


