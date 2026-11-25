"""Relay admission guards, retry/cooldown recovery, cancellation tombstones, and manual retry."""

from __future__ import annotations

import hashlib
import time
from typing import NamedTuple

from .continuity import MAIN as MAIN_CALL

CODEX_PROFILE_NAME = "codex"
DEFAULT_MAX_RETRIES = 12  # Pinned RoboProbe joint-policy default; excludes the first attempt.
RETRY_POLICY = "bounded_harness_retry_after_explicit_retryable_provider_error"
TRANSPORT_RECOVERY_KEYS = frozenset((
    "max_rounds",
    "initial_delay_s",
    "backoff_multiplier",
    "max_delay_s",
    "max_episode_delay_s",
    "transport_incomplete",
))


class IdleRun(NamedTuple):
    """Consume-guard state of one main thread (replaced on every delivered usable main completion).

    items: number of conversation items of that delivered request (``history_fingerprints`` length).
    last_blocks: number of content blocks of its last item (0 when it had no item).
    idle: how many delivered main completions in a row, ending with this one, were requests that added
        no client input to the history of the delivered request before them.
    """

    items: int
    last_blocks: int
    idle: int


class ContinuityAdmission(NamedTuple):
    """``ModelRequestProxy._observe_continuity`` result for one actor generation request.

    call_purpose: ``"main"``, ``"helper"`` or ``"subagent"``, committed with the request.
    idle: consume-guard verdict of a compared main request (True/False/None).
    refused: consecutive idle delivered completions count when the consume guard refuses this request.
    """

    call_purpose: str
    idle: bool | None
    refused: int | None


class GuardMixin:
    """Budget/retry cooldown, client runaway & consume guards, and thread tombstones sharing ``self.lock``."""

    def cancel_pending(self, reason="episode_stopped", *, threads=()):
        """Cancel every in-flight model call and tombstone threads on non-Codex profiles."""
        tombstoned = []
        with self.lock:
            active = list(self.active.values())
            if self.wire.client != CODEX_PROFILE_NAME:
                candidates = [state["identity"].get("native_thread_id") for state in active] + list(threads)
                for thread_id in candidates:
                    if isinstance(thread_id, str) and thread_id and thread_id not in self.cancelled_threads:
                        self.cancelled_threads[thread_id] = reason
                        self._reopenable_threads.add(thread_id)
                        tombstoned.append(thread_id)
        for state in active:
            self._cancel(state, reason)
        for thread_id in tombstoned:
            self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason, reopenable=True)

    def block_threads(self, threads, reason):
        """Put reopenable ``cancel_pending`` tombstones on ``threads`` (non-Codex profiles only)."""
        if self.wire.client == CODEX_PROFILE_NAME:
            return []
        tombstoned = []
        with self.lock:
            for thread_id in threads:
                if isinstance(thread_id, str) and thread_id and thread_id not in self.cancelled_threads:
                    self.cancelled_threads[thread_id] = reason
                    self._reopenable_threads.add(thread_id)
                    tombstoned.append(thread_id)
            active = [
                state
                for state in self.active.values()
                if state["identity"].get("native_thread_id") in tombstoned
            ]
        for state in active:
            self._cancel(state, reason)
        for thread_id in tombstoned:
            self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason, reopenable=True)
        return tombstoned

    def reopen_thread(self, thread_id):
        """Lift a reopenable ``cancel_pending`` tombstone on ``thread_id``."""
        with self.lock:
            if thread_id not in self._reopenable_threads:
                return False
            self._reopenable_threads.discard(thread_id)
            reason = self.cancelled_threads.pop(thread_id, None)
        self._record("model_thread_reopened", native_thread_id=thread_id, cancel_reason=reason)
        return True

    def register_thread(self, thread_id, source, request_id):
        """Attribute an official native thread and lift any reopenable tombstone on it."""
        if any(
            not isinstance(value, str) or not value or len(value) > limit or "\n" in value or "\r" in value
            for value, limit in ((thread_id, 256), (source, 128), (request_id, 128))
        ):
            raise ValueError("model_thread_registration_invalid")
        owner = dict(source=source, caller_request_id=request_id)
        self.reopen_thread(thread_id)
        with self.lock:
            if thread_id in self.cancelled_threads:
                raise ValueError("model_thread_already_cancelled")
            if thread_id in self.thread_owners and self.thread_owners[thread_id] != owner:
                raise ValueError("model_thread_owner_conflict")
            self.thread_owners[thread_id] = owner
        self._record("model_thread_registered", native_thread_id=thread_id, **owner)

    def unregister_thread(self, thread_id):
        """Remove thread owner attribution while preserving any cancellation tombstone."""
        with self.lock:
            owner = self.thread_owners.pop(thread_id, None)
        if owner is not None:
            self._record("model_thread_unregistered", native_thread_id=thread_id, **owner)

    def cancel_thread(self, thread_id, reason="helper_turn_interrupted"):
        """Cancel ``thread_id`` permanently and reject its future HTTP admissions."""
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("model_thread_cancellation_invalid")
        with self.lock:
            self.cancelled_threads.setdefault(thread_id, reason)
            self._reopenable_threads.discard(thread_id)
            for state in list(self.active.values()):
                if state["identity"].get("native_thread_id") == thread_id:
                    self._cancel(state, reason)
        self._record("model_thread_cancelled", native_thread_id=thread_id, reason=reason)

    @property
    def transport_blocked(self):
        with self.lock:
            return self.blocked_error is not None

    @property
    def pending_requests(self):
        with self.lock:
            return bool(self.active)

    @property
    def last_activity_at(self):
        """Latest actual request/transport activity on the monotonic clock, or None."""
        with self.lock:
            return self._last_activity_at

    @property
    def model_wait_seconds(self):
        """Union of intervals with any pending logical request, including now."""
        with self.lock:
            return self._wait_total_s + (
                time.monotonic() - self._wait_started_at if self._wait_started_at is not None else 0.0
            )

    @property
    def transport_recovery_summary(self):
        with self.lock:
            return dict(
                policy=dict(self.transport_recovery) if self.transport_recovery else None,
                rounds=self._recovery_rounds,
                recovered_calls=self._recovered_calls,
                denials=self._recovery_denials,
                cooldown_s=self._recovery_delay_s,
            )

    def _recovery_delay(self, state, rounds, explicit_retryable):
        """Return next cooldown delay in seconds before resending, or None if not eligible."""
        policy = self.transport_recovery
        evidence = state.get("failure_evidence") or {}
        if policy is None or not (
            explicit_retryable
            or evidence.get("classification") == "quota_exceeded"
            or (policy["transport_incomplete"] and evidence.get("classification") == "transport_incomplete")
        ):
            return None
        delay = policy["initial_delay_s"]
        for _ in range(rounds):
            delay = min(policy["max_delay_s"], delay * policy["backoff_multiplier"])
        delay = min(delay, policy["max_delay_s"])
        reason = "recovery_rounds_exhausted" if rounds >= policy["max_rounds"] else self._recovery_gate_reason()
        if reason is None:
            with self.lock:
                if self._recovery_delay_s + delay > policy["max_episode_delay_s"]:
                    reason = "episode_recovery_budget_exhausted"
                else:
                    self._recovery_delay_s += delay
                    self._recovery_rounds += 1
        if reason is not None:
            self._deny_recovery(state, reason, rounds)
            return None
        return delay

    def _recovery_gate_reason(self):
        with self.lock:
            if self.closed:
                return "proxy_closed"
            if self.blocked_error is not None:
                return "transport_blocked"
        if self.recovery_gate is None:
            return "recovery_gate_unavailable"
        try:
            reason = self.recovery_gate()
        except Exception as error:
            return "recovery_gate_error:" + type(error).__name__
        return reason if reason is None or isinstance(reason, str) else "recovery_gate_invalid"

    def _deny_recovery(self, state, reason, rounds):
        with self.lock:
            self._recovery_denials += 1
        self._record(
            "model_transport_recovery_denied",
            **state["identity"],
            reason=reason,
            recovery_rounds=rounds,
            error=state.get("failure_evidence"),
        )

    def _block_after_failure(self, state, error_type):
        if state["cancel"].is_set() and state.get("cancel_reason") != "client_disconnected":
            return
        error = dict(
            logical_call_id=state["identity"]["logical_call_id"],
            error_type=error_type,
            manual_retry_allowed=bool(state.get("provider_explicit_error")),
            evidence=state.get("failure_evidence"),
        )
        with self.lock:
            if self.blocked_error is not None:
                return
            self.blocked_error = error
        self._record(
            "model_transport_blocked",
            **state["identity"],
            error=error,
            automatic_retries=self.max_retries,
            manual_retry_allowed=error["manual_retry_allowed"],
        )

    def _runaway_admission(self, thread_id, body):
        """Count identical consecutive generation bodies on ``thread_id`` within ``RUNAWAY_WINDOW_S``."""
        digest = hashlib.sha256(body).digest()
        now = time.monotonic()
        with self.lock:
            previous, times = self._runaway_runs.get(thread_id, (None, ()))
            times = [t for t in times if now - t <= self.RUNAWAY_WINDOW_S] if previous == digest else []
            times.append(now)
            self._runaway_runs[thread_id] = (digest, times)
            return len(times) if len(times) > self.RUNAWAY_MAX_IDENTICAL_ADMISSIONS else None

    def _block_client_loop(self, state, classification, error_type, **facts):
        """Block the transport after a client re-request loop while allowing operator manual retry."""
        evidence = dict(classification=classification, **facts)
        state["failure_evidence"] = evidence
        error = dict(
            logical_call_id=state["identity"]["logical_call_id"],
            error_type=error_type,
            manual_retry_allowed=True,
            evidence=evidence,
        )
        with self.lock:
            first = self.blocked_error is None
            if first:
                self.blocked_error = error
        self._record("model_" + classification, **state["identity"], **facts)
        if first:
            self._record(
                "model_transport_blocked",
                **state["identity"],
                error=error,
                automatic_retries=self.max_retries,
                manual_retry_allowed=True,
            )

    def _block_runaway(self, state, count):
        """Block transport after tight identical-body retry loop."""
        self._block_client_loop(
            state,
            "client_retry_runaway",
            "ClientRetryRunaway",
            identical_admissions=count,
            window_s=self.RUNAWAY_WINDOW_S,
        )

    def _block_unconsumed(self, state, idle):
        """Block transport after repeated idle delivered completions that the client did not act on."""
        self._block_client_loop(
            state,
            "client_did_not_consume_completion",
            "ClientDidNotConsumeCompletion",
            idle_completions=idle,
            max_idle_completions=self.CONSUME_GUARD_MAX_IDLE_COMPLETIONS,
        )

    def _idle_request(self, thread, admission, fingerprints, output_flags):
        """Consume-guard verdict under ``self.lock`` for an admitted main request."""
        run = self._idle_runs.get(thread)
        if run is None or admission.call_purpose != MAIN_CALL or len(output_flags) != len(fingerprints):
            return None
        if admission.reason is not None or len(fingerprints) < run.items:
            return False
        n = run.items
        if not all(output_flags[n:]):
            return False
        return not (n and len(fingerprints[n - 1][1]) > run.last_blocks and not output_flags[n - 1])

    def _commit_idle_run(self, thread, call_purpose, fingerprints, idle):
        """Replace ``thread``'s ``IdleRun`` with a delivered usable main request (under ``self.lock``)."""
        if call_purpose != MAIN_CALL:
            return
        previous = self._idle_runs.get(thread)
        count = previous.idle + 1 if idle and previous is not None else 0
        self._idle_runs[thread] = IdleRun(len(fingerprints), len(fingerprints[-1][1]) if fingerprints else 0, count)

    def manual_retry(self):
        """Authorize one operator manual retry after an explicit provider or client-loop block."""
        with self.lock:
            if self.active:
                raise ValueError("manual_retry_requires_no_active_model_calls")
            if not self.blocked_error or not self.blocked_error["manual_retry_allowed"]:
                raise ValueError("manual_retry_requires_explicit_provider_error")
            previous = self.blocked_error
            self.blocked_error = None
            self._runaway_runs.clear()
            self._idle_runs.clear()
        self._record("model_manual_retry_authorized", previous_error=previous)
