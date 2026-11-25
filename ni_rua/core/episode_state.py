"""Typed run- and episode-scoped state ledger replacing ad-hoc attribute stashing on Harness."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")
SlotScope = Literal["run", "episode"]

_FALLBACK_LEDGER_ATTR = "_ni_rua_episode_state_fallback"


def _normalize_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        return {}
    return dict(identity)


def _unwrap_owner(owner: Any) -> Any:
    while owner is not None and not isinstance(getattr(owner, "episode_state", None), EpisodeStateLedger):
        inner_dict = getattr(owner, "__dict__", None)
        if isinstance(inner_dict, dict) and "_runtime" in inner_dict:
            owner = inner_dict["_runtime"]
        else:
            break
    return owner


@dataclass(frozen=True)
class StateSlot(Generic[T]):
    """Typed key into :class:`EpisodeStateLedger` with explicit ``"run"`` or ``"episode"`` lifecycle."""

    name: str
    default_factory: Callable[[], T]
    scope: SlotScope = "episode"


class EpisodeStateLedger:
    """Centralized run/episode state ledger and causal observation-basis tracker."""

    def __init__(self, identity: Mapping[str, Any] | None = None) -> None:
        self._identity: dict[str, Any] = _normalize_identity(identity)
        self._run_store: dict[str, Any] = {}
        self._episode_store: dict[str, Any] = {}
        self.motion_epoch: int = 0
        self._seq: int = 0
        self._continuity_floor_seq: int = 0
        self._obs_seq: dict[str, int] = {}
        self.current_observation_id: str | None = None
        self.active_action_id: str | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def _store_for(self, slot: StateSlot[Any]) -> dict[str, Any]:
        if slot.scope == "run":
            return self._run_store
        return self._episode_store

    def sync_identity(self, identity: Mapping[str, Any] | None) -> bool:
        """Synchronize against ``identity``; automatically resets episode/run slots if identity changed."""
        normalized = _normalize_identity(identity)
        if not normalized:
            return False
        if not self._identity:
            self._identity = normalized
            return False
        if normalized == self._identity:
            return False
        prev_run = self._identity.get("run_id")
        next_run = normalized.get("run_id")
        if prev_run is not None and next_run is not None and prev_run != next_run:
            self.begin(normalized)
        else:
            self.begin_episode(normalized)
        return True

    def begin(self, identity: Mapping[str, Any] | None = None) -> None:
        """Start or restart a run, clearing both run- and episode-scoped slots."""
        if identity is not None:
            self._identity = _normalize_identity(identity)
        self._run_store.clear()
        self._episode_store.clear()
        self.motion_epoch += 1
        self._seq = 0
        self._continuity_floor_seq = 0
        self._obs_seq.clear()
        self.current_observation_id = None
        self.active_action_id = None

    def begin_episode(self, identity: Mapping[str, Any] | None = None) -> None:
        """Advance to a new episode/generation, clearing episode-scoped slots while retaining run slots."""
        if identity is not None:
            self._identity = _normalize_identity(identity)
        self._episode_store.clear()
        self.motion_epoch += 1
        self._obs_seq.clear()
        self._continuity_floor_seq = self._seq + 1
        self.current_observation_id = None
        self.active_action_id = None

    def get(self, slot: StateSlot[T]) -> T:
        store = self._store_for(slot)
        if slot.name not in store:
            store[slot.name] = slot.default_factory()
        return store[slot.name]

    def peek(self, slot: StateSlot[T], default: Any = None) -> Any:
        return self._store_for(slot).get(slot.name, default)

    def set(self, slot: StateSlot[T], value: T) -> T:
        self._store_for(slot)[slot.name] = value
        return value

    def has(self, slot: StateSlot[Any]) -> bool:
        return slot.name in self._store_for(slot)

    def reset_slot(self, slot: StateSlot[Any]) -> None:
        self._store_for(slot).pop(slot.name, None)

    def observation_published(self, observation_id: str | None, *, make_current: bool = True) -> int:
        """Record a published observation and return its monotonic sequence index."""
        self._seq += 1
        if observation_id:
            obs_key = str(observation_id)
            self._obs_seq[obs_key] = self._seq
            if make_current:
                self.current_observation_id = obs_key
        return self._seq

    def action_admitted(self, action_id: str | None = None) -> int:
        """Mark physical action admission and advance ``motion_epoch``."""
        self.active_action_id = str(action_id) if action_id else None
        self.motion_epoch += 1
        return self.motion_epoch

    def action_settled(
        self,
        *,
        continuous: bool = True,
        terminal_observation_id: str | None = None,
    ) -> int:
        """Mark physical action completion; invalidate pre-action delta bases when ``not continuous``."""
        self.active_action_id = None
        self.motion_epoch += 1
        if not continuous:
            self._continuity_floor_seq = self._seq + 1
            self.current_observation_id = None
        elif terminal_observation_id:
            obs_key = str(terminal_observation_id)
            if obs_key not in self._obs_seq or self._obs_seq[obs_key] < self._continuity_floor_seq:
                self.observation_published(obs_key, make_current=True)
            else:
                self.current_observation_id = obs_key
        return self.motion_epoch

    def is_basis(self, observation_id: str | None) -> bool:
        """Return ``True`` iff ``observation_id`` remains causally connected across settled actions."""
        if not observation_id:
            return False
        obs_key = str(observation_id)
        seq = self._obs_seq.get(obs_key)
        if seq is not None:
            return seq >= self._continuity_floor_seq
        return self._continuity_floor_seq == 0


def episode_state_of(owner: Any, *, create_fallback: bool = True) -> EpisodeStateLedger | None:
    """Resolve ``owner.episode_state`` (or attach an internal ledger on lightweight test doubles)."""
    owner = _unwrap_owner(owner)
    if owner is None:
        return None
    ledger = getattr(owner, "episode_state", None)
    if isinstance(ledger, EpisodeStateLedger):
        ledger.sync_identity(getattr(owner, "identity", None))
        return ledger
    owner_dict = getattr(owner, "__dict__", None)
    if isinstance(owner_dict, dict):
        fallback = owner_dict.get(_FALLBACK_LEDGER_ATTR)
        if isinstance(fallback, EpisodeStateLedger):
            fallback.sync_identity(getattr(owner, "identity", None))
            return fallback
        if create_fallback:
            fallback = EpisodeStateLedger(getattr(owner, "identity", None))
            owner_dict[_FALLBACK_LEDGER_ATTR] = fallback
            return fallback
    return None


def slot_get(owner: Any, slot: StateSlot[T], *, legacy_attr: str | None = None) -> T:
    """Read ``slot`` from ``owner.episode_state``; preserve ``legacy_attr`` on lightweight test doubles."""
    owner = _unwrap_owner(owner)
    if owner is None:
        return slot.default_factory()
    if not isinstance(getattr(owner, "episode_state", None), EpisodeStateLedger) and legacy_attr:
        owner_dict = getattr(owner, "__dict__", None)
        if isinstance(owner_dict, dict):
            if legacy_attr not in owner_dict:
                owner_dict[legacy_attr] = slot.default_factory()
            return owner_dict[legacy_attr]
    ledger = episode_state_of(owner)
    if ledger is None:
        return slot.default_factory()
    return ledger.get(slot)


def slot_peek(owner: Any, slot: StateSlot[T], default: Any = None, *, legacy_attr: str | None = None) -> Any:
    """Peek ``slot`` without populating ``default_factory``."""
    owner = _unwrap_owner(owner)
    if owner is None:
        return default
    if not isinstance(getattr(owner, "episode_state", None), EpisodeStateLedger) and legacy_attr:
        owner_dict = getattr(owner, "__dict__", None)
        if isinstance(owner_dict, dict) and legacy_attr in owner_dict:
            return owner_dict[legacy_attr]
    ledger = episode_state_of(owner, create_fallback=False)
    if ledger is None:
        return default
    return ledger.peek(slot, default)


def slot_set(owner: Any, slot: StateSlot[T], value: T, *, legacy_attr: str | None = None) -> T:
    """Store ``value`` in ``owner.episode_state`` (or ``legacy_attr`` on lightweight test doubles)."""
    owner = _unwrap_owner(owner)
    if owner is None:
        return value
    if not isinstance(getattr(owner, "episode_state", None), EpisodeStateLedger) and legacy_attr:
        owner_dict = getattr(owner, "__dict__", None)
        if isinstance(owner_dict, dict):
            owner_dict[legacy_attr] = value
            return value
    ledger = episode_state_of(owner)
    if ledger is not None:
        ledger.set(slot, value)
    return value


def slot_reset(owner: Any, slot: StateSlot[Any], *, legacy_attr: str | None = None) -> None:
    """Reset ``slot`` in ``owner.episode_state``."""
    owner = _unwrap_owner(owner)
    if owner is None:
        return
    if not isinstance(getattr(owner, "episode_state", None), EpisodeStateLedger) and legacy_attr:
        owner_dict = getattr(owner, "__dict__", None)
        if isinstance(owner_dict, dict):
            owner_dict.pop(legacy_attr, None)
    ledger = episode_state_of(owner, create_fallback=False)
    if ledger is not None:
        ledger.reset_slot(slot)
