"""Dependency-free client for a preconnected, invocation-owned program channel."""
import base64
import io
import json
import os
import threading

def _is_zero_step_not_executed_plan(response):
    """判定响应是否为具备明确零步未执行证据（规划失败或 execute_if_clean 降级）的计划返回。"""
    if not isinstance(response, dict) or response.get("kind") != "plan" or response.get("success") is False:
        return False
    data = response.get("data")
    if not isinstance(data, dict):
        return False
    if data.get("physics_advanced") is not False:
        return False
    if type(data.get("executed_native_steps")) is not int or data["executed_native_steps"] != 0:
        return False
    status = data.get("status")
    if status == "failed":
        return True
    return status == "planned" and data.get("downgraded_from_execute_if_clean") is True


class ActionIncompleteError(RuntimeError):
    """当 require_completed=True 且动作未达已证实物理 completed 状态时抛出的异常。"""

    def __init__(self, response):
        is_dict = isinstance(response, dict)
        kind = response.get("kind") if is_dict else None
        data = response.get("data") if is_dict and isinstance(response.get("data"), dict) else None
        self.response = response
        self.plan_id = data.get("plan_id") if isinstance(data, dict) else None
        if kind == "action":
            receipt = data.get("receipt") if isinstance(data, dict) else None
            self.receipt = dict(receipt) if isinstance(receipt, dict) else {}
            status = self.receipt.get("status")
            if not isinstance(status, str) or not status:
                status = "unknown"
        elif _is_zero_step_not_executed_plan(response):
            raw_receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
            downgrade_reasons = (
                list(data["downgrade_reasons"])
                if isinstance(data.get("downgrade_reasons"), list)
                else None
            )
            reason_val = (
                data.get("reason")
                or raw_receipt.get("reason")
                or (downgrade_reasons[0] if downgrade_reasons else None)
                or ("downgraded_from_execute_if_clean" if data.get("downgraded_from_execute_if_clean") is True else None)
            )
            self.receipt = dict(raw_receipt)
            self.receipt.update(status="not_executed", physics_advanced=False, executed_native_steps=0)
            if reason_val is not None:
                self.receipt["reason"] = reason_val
            if downgrade_reasons is not None:
                self.receipt["downgrade_reasons"] = downgrade_reasons
            if "downgraded_from_execute_if_clean" in data:
                self.receipt["downgraded_from_execute_if_clean"] = data["downgraded_from_execute_if_clean"]
            if data.get("failed_stage_id") and "failed_stage_id" not in self.receipt:
                self.receipt["failed_stage_id"] = data["failed_stage_id"]
            if "blocking_risks" in data and "blocking_risks" not in self.receipt:
                self.receipt["blocking_risks"] = data["blocking_risks"]
            status = "not_executed"
        else:
            self.receipt = {}
            status = "unknown"
        if isinstance(data, dict) and "next_action" in data and "next_action" not in self.receipt:
            self.receipt["next_action"] = data["next_action"]
        if self.plan_id and "plan_id" not in self.receipt:
            self.receipt["plan_id"] = self.plan_id
        reason = self.receipt.get("reason")
        failed_stage = self.receipt.get("failed_stage_id")
        next_action = self.receipt.get("next_action") if isinstance(self.receipt.get("next_action"), dict) else {}
        start_from = next_action.get("start_from_stage_id")
        parts = [f"action_incomplete: status={status}"]
        if reason:
            parts.append(f"reason={reason}")
        if self.receipt.get("downgrade_reasons"):
            parts.append(f"downgrade_reasons={self.receipt['downgrade_reasons']}")
        if self.plan_id:
            parts.append(f"plan_id={self.plan_id}")
        if failed_stage:
            parts.append(f"failed_stage_id={failed_stage}")
        if start_from and start_from != failed_stage:
            parts.append(f"start_from_stage_id={start_from}")
        super().__init__(", ".join(parts))


class ProgramBrokerError(RuntimeError):
    """Typed broker-side RPC error preserving structured fields (e.g. incomplete_action)."""

    def __init__(self, error):
        if isinstance(error, dict):
            self.payload = dict(error)
            self.code = error.get("code")
            self.incomplete_action = error.get("incomplete_action")
            msg = error.get("message") or error.get("code") or json.dumps(error)
            if self.code and not str(msg).startswith(str(self.code)):
                msg = f"{self.code}: {msg}"
            super().__init__(msg)
        else:
            self.payload = None
            self.code = None
            self.incomplete_action = None
            super().__init__(error)


class NiRuaClient:
    def __init__(self):
        raw = os.fdopen(os.dup(int(os.environ["NI_RUA_PROGRAM_FD"])),"r+b",buffering=0)
        self.pipe = io.BufferedRWPair(raw, raw)
        self.sequence=0
        self.lock=threading.RLock()
        self._failure_acknowledged=False
        self._unacknowledged_failed_action=None
        self._last_response_physical=False

    def acknowledge_failure(self):
        """显式确认已知悉上一次未完成物理动作，允许下一次物理动作继续发起。"""
        self._failure_acknowledged=True
        self._unacknowledged_failed_action=None

    @staticmethod
    def ensure_completed(response):
        """断言动作响应已完成（要求 kind=='action' 且 receipt.status=='completed'），否则抛出 ActionIncompleteError。"""
        if isinstance(response, dict) and response.get("kind") == "action":
            data = response.get("data")
            receipt = data.get("receipt") if isinstance(data, dict) else None
            if isinstance(receipt, dict) and receipt.get("status") == "completed":
                return response
        raise ActionIncompleteError(response)

    def _request(self,value):
        with self.lock:
            self.pipe.write((json.dumps(value)+"\n").encode())
            self.pipe.flush()
            line=self.pipe.readline(32*1024*1024)
        if not line: raise RuntimeError("program_channel_closed")
        response=json.loads(line)
        self._last_response_physical = bool(response.get("physical_action"))
        if not response["ok"]: raise ProgramBrokerError(response["error"])
        if "incomplete_action_latched" in response:
            self._unacknowledged_failed_action = response["incomplete_action_latched"]
        return response["result"]

    def call(self,tool_name,arguments=None,*,allow_after_failure:bool=False):
        """Return success/kind/data plus display contentItems. For kind='action',
        read data['receipt']; success alone never establishes physical success.
        Old display-only replies have no typed evidence and remain unknown.
        """
        # A review may keep this RPC suspended while the same Python stack and
        # local variables are preserved. Only the main actor can resume it.
        args = dict(arguments or {})
        allow_from_args = bool(args.pop("allow_after_failure", False))
        effective_allow = bool(allow_after_failure or allow_from_args or self._failure_acknowledged)
        is_physical_hint = (
            (tool_name == "robot_action" and (
                args.get("mode", "preview") in ("execute", "direct", "execute_if_clean")
                or bool(args.get("auto_execute_if_clean"))
                or ("mode" not in args and "operation" in args and args.get("operation") not in ("preview", "preflight", "inspect", "reach"))
            ))
            or tool_name in ("robot_wait", "robot_policy_step", "robot_move", "robot_grasp", "robot_gripper", "robot_batch")
        )
        with self.lock:
            self.sequence+=1
            self._last_response_physical = False
            frame = {
                "operation_id": str(self.sequence),
                "tool": tool_name,
                "arguments": args,
                "allow_after_failure": effective_allow,
            }
            try:
                result = self._request(frame)
            except Exception:
                if (is_physical_hint or self._last_response_physical) and self._failure_acknowledged:
                    self._failure_acknowledged = False
                raise
            if (
                self._last_response_physical
                or is_physical_hint
                or (isinstance(result, dict) and result.get("kind") == "action")
            ) and self._failure_acknowledged:
                self._failure_acknowledged = False
        return result

    def observe(self, **arguments):
        """采集或检视当前允许观测（转发至 robot_observe）。"""
        return self.call("robot_observe", arguments)

    def action(self, stages=None, *, observation_id=None, require_completed=False, allow_after_failure: bool = False, **arguments):
        """Request execution; the broker previews and yields for main-actor review
        unless that actor authorized direct actions when starting this program.
        When require_completed=True, raises ActionIncompleteError if receipt.status != 'completed'.
        """
        if isinstance(stages, dict):
            payload = dict(stages)
            payload.update(arguments)
            payload.setdefault('mode', 'direct')
            if observation_id is not None:
                payload['observation_id'] = observation_id
        else:
            if observation_id is None and arguments.get('mode', 'direct') != 'execute':
                raise TypeError("NiRuaClient.action() missing 1 required keyword-only argument: 'observation_id'")
            payload = dict(arguments, mode=arguments.get('mode', 'direct'))
            if observation_id is not None:
                payload['observation_id'] = observation_id
            if stages is not None:
                payload['stages'] = stages
        result = self.call('robot_action', payload, allow_after_failure=allow_after_failure)
        if require_completed:
            self.ensure_completed(result)
        return result

    def artifact(self,reference):
        return base64.b64decode(self._request({"artifact_ref":reference})["data_base64"])

    def stereo_guide(self, *, observation_id, source_camera_id, target_camera_id, source_pixel):
        """只读双目极线约束查询辅助方法。"""
        return self.call("stereo_guide", dict(
            observation_id=observation_id,
            source_camera_id=source_camera_id,
            target_camera_id=target_camera_id,
            source_pixel=list(source_pixel),
        ))
