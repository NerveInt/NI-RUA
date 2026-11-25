"""Read-only run metadata, SQLite evidence, and multi-backend usage/cost resolution for SO-101 videos.

All readers are strictly observer-only and sidecar-free (sealed WAL archives are
opened through ``open_read_store``). Missing metadata stays ``None`` and is never
guessed or fabricated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

BACKEND_LABELS = {
    "opencode": "OpenCode",
    "codex": "Codex",
    "claude_code": "CC",
    "cc": "CC",
    "external": "External",
}
MODEL_TOKEN_LABELS = {
    "deepseek": "DeepSeek",
    "glm": "GLM",
    "gemini": "Gemini",
    "kimi": "Kimi",
    "qwen": "Qwen",
    "gpt": "GPT",
    "claude": "Claude",
}

_RUN_MARKERS = ("effective_config.json", "events.sqlite", "agent-raw.jsonl", "summary.json", "runtime.json")


def _finite_nonneg_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _nonneg_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _read_json_dict(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _is_run_root(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    for name in _RUN_MARKERS:
        marker = candidate / name
        if marker.is_file() and not marker.is_symlink():
            return True
    manifest = _read_json_dict(candidate / "manifest.json")
    return bool(manifest and ("run_id" in manifest or "episode_id" in manifest or "effective_config" in manifest))


def find_run_root(directory: Path | None) -> Path | None:
    """Locate the enclosing NI-RUA run directory for a recording directory.

    Checks bounded ancestors ``directory.parents[1:5]`` covering:
    ``<run>/videos/<id>``, ``<run>/robot/videos/<id>``, and
    ``<run>/private/robot/videos/<id>``.
    """
    if directory is None:
        return None
    for ancestor in directory.parents[1:5]:
        if _is_run_root(ancestor):
            return ancestor
    return None


def _file_sig(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size)


def _decode_sqlite_snapshot(conn: sqlite3.Connection) -> tuple[list[dict], dict[str, dict]]:
    events: list[dict] = []
    runtime_state: dict[str, dict] = {}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if "events" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        if "body" in columns:
            order = " ORDER BY seq" if "seq" in columns else ""
            for (body,) in conn.execute(f"SELECT body FROM events{order}").fetchall():
                if not isinstance(body, str) or not body:
                    continue
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
        elif {"type", "occurred_at"} <= columns:
            extra = ", payload" if "payload" in columns else ""
            for row in conn.execute(f"SELECT type, occurred_at{extra} FROM events").fetchall():
                payload = {}
                if len(row) > 2 and isinstance(row[2], str) and row[2]:
                    try:
                        loaded = json.loads(row[2])
                        if isinstance(loaded, dict):
                            payload = loaded
                    except json.JSONDecodeError:
                        pass
                events.append({"type": row[0], "occurred_at": row[1], "payload": payload})

    if "state" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(state)").fetchall()}
        if {"kind", "id", "body"} <= columns:
            rows = conn.execute(
                "SELECT id, body FROM state WHERE kind='runtime' AND id IN ('token_accounting', 'token_usage')"
            ).fetchall()
            for state_id, body in rows:
                if not isinstance(body, str) or not body:
                    continue
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    runtime_state[str(state_id)] = parsed

    return events, runtime_state


def read_sqlite_snapshot(run_root: Path | None) -> tuple[list[dict], dict[str, dict]]:
    """Read ``(events, runtime_state)`` from ``<run_root>/events.sqlite`` strictly read-only without sidecars."""
    if run_root is None:
        return [], {}
    database = run_root / "events.sqlite"
    wal = run_root / "events.sqlite-wal"
    if not database.is_file() or database.is_symlink() or wal.is_symlink():
        return [], {}
    try:
        from ni_rua.core.storage.store import open_read_store
    except Exception:
        open_read_store = None

    if open_read_store is not None:
        for _ in range(3):
            try:
                with open_read_store(run_root) as store:
                    return _decode_sqlite_snapshot(store.db)
            except ValueError as err:
                if str(err) == "archived_database_changed_during_snapshot":
                    continue
                break
            except (sqlite3.Error, OSError):
                break
        return [], {}

    # Sidecar-free read-only fallback for standalone SQLite fixtures.
    if wal.exists():
        for _ in range(3):
            try:
                before = {path: _file_sig(path) for path in (database, wal)}
                with tempfile.TemporaryDirectory(prefix="ni-rua-video-events-") as temp_dir:
                    copy_root = Path(temp_dir)
                    for path in before:
                        shutil.copyfile(path, copy_root / path.name)
                    if {path: _file_sig(path) for path in (database, wal)} != before:
                        continue
                    uri = (copy_root / "events.sqlite").resolve().as_uri() + "?mode=ro"
                    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
                    try:
                        conn.execute("PRAGMA query_only=ON")
                        return _decode_sqlite_snapshot(conn)
                    finally:
                        conn.close()
            except (sqlite3.Error, OSError):
                break
        return [], {}

    uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            conn.execute("PRAGMA query_only=ON")
            return _decode_sqlite_snapshot(conn)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return [], {}


def backend_label(backend: str) -> str:
    cleaned = str(backend).strip()
    return BACKEND_LABELS.get(cleaned.lower(), cleaned)


def model_label(model: str) -> str:
    """Prettify a raw model id, e.g. ``deepseek-v4.1-flash`` -> ``DeepSeek 4.1 Flash``."""
    raw = str(model).strip()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1].strip()
    words: list[str] = []
    for token in raw.split("-"):
        if not token:
            continue
        low = token.lower()
        if len(low) > 1 and low[0] == "v" and low[1].isdigit():
            words.append(low[1:])
            continue
        prefix = next((k for k in MODEL_TOKEN_LABELS if low.startswith(k)), None)
        if prefix is not None:
            words.append(MODEL_TOKEN_LABELS[prefix] + low[len(prefix):])
        elif low[:1].isdigit():
            words.append(low)
        else:
            words.append(low[:1].upper() + low[1:])
    return " ".join(words) or raw


def _extract_identity_from_config(config: dict, source: str) -> tuple[str | None, str | None, str | None, str | None]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    model_name = model_cfg.get("name")
    if isinstance(model_name, str) and model_name.strip():
        # Config.to_dict() omits ``agent`` when it equals default AgentConfig(backend="codex").
        raw_backend = agent_cfg.get("backend")
        backend = raw_backend.strip() if isinstance(raw_backend, str) and raw_backend.strip() else "codex"
        raw_effort = model_cfg.get("effort")
        effort = raw_effort.strip() if isinstance(raw_effort, str) and raw_effort.strip() else None
        return backend, model_name.strip(), effort, source
    return None, None, None, None


def _config_identity(
    run_root: Path | None,
    *,
    config_dict: dict | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(backend, model_name, effort, source)`` from ``effective_config.json``, ``manifest.json``, or ``config_dict``."""
    if run_root is not None:
        config = _read_json_dict(run_root / "effective_config.json")
        if config is not None:
            found = _extract_identity_from_config(config, "effective_config.json")
            if found[1] is not None:
                return found

        manifest = _read_json_dict(run_root / "manifest.json")
        if manifest is not None:
            embedded = manifest.get("effective_config") if isinstance(manifest.get("effective_config"), dict) else {}
            model_cfg = embedded.get("model") if isinstance(embedded.get("model"), dict) else {}
            agent_cfg = embedded.get("agent") if isinstance(embedded.get("agent"), dict) else {}
            model_name = model_cfg.get("name") or manifest.get("model")
            if isinstance(model_name, str) and model_name.strip():
                raw_backend = agent_cfg.get("backend") or manifest.get("agent_backend")
                backend = raw_backend.strip() if isinstance(raw_backend, str) and raw_backend.strip() else "codex"
                raw_effort = model_cfg.get("effort") or manifest.get("effort")
                effort = raw_effort.strip() if isinstance(raw_effort, str) and raw_effort.strip() else None
                return backend, model_name.strip(), effort, "manifest.json"

    if isinstance(config_dict, dict):
        found = _extract_identity_from_config(config_dict, "runtime_config")
        if found[1] is not None:
            return found

    return None, None, None, None


def resolve_run_label(
    run_root: Path | None,
    *,
    model_override: str | None = None,
    label_override: str | None = None,
    config_dict: dict | None = None,
) -> dict | None:
    """Resolve the header left label ``Backend-Model-Effort`` and its source fields."""
    backend, model_name, effort, source = _config_identity(run_root, config_dict=config_dict)
    if label_override is not None and str(label_override).strip():
        return {
            "label": str(label_override).strip(),
            "backend": backend,
            "model": model_name,
            "effort": effort,
            "source": "explicit_label_override",
        }
    clean_model_override = str(model_override).strip() if isinstance(model_override, str) and model_override.strip() else None
    if model_name is None and clean_model_override is None:
        return None
    parts: list[str] = []
    if backend:
        parts.append(backend_label(backend))
    parts.append(clean_model_override or model_label(model_name or ""))
    if effort:
        parts.append(effort)
    return {
        "label": "-".join(parts),
        "backend": backend,
        "model": model_name,
        "effort": effort,
        "source": source or "explicit_model_override",
    }


def _usage_from_agent_raw(run_root: Path | None) -> dict:
    """Extract cumulative tokens & provider-reported cost from ``agent-raw.jsonl`` (OpenCode / Claude Code)."""
    empty = dict(total_tokens=None, cost_usd=None, currency=None, tokens_source=None, cost_source=None)
    if run_root is None:
        return empty
    raw_path = run_root / "agent-raw.jsonl"
    if not raw_path.is_file() or raw_path.is_symlink():
        return empty

    opencode_sessions: dict[str, dict] = {}
    claude_results: dict[str, dict] = {}
    try:
        with raw_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                # Fast substring pre-check skips megabytes of base64 MCP tool results without JSON parsing.
                has_opencode = '"session.usage.updated"' in line
                has_claude = '"ResultMessage"' in line
                if not has_opencode and not has_claude:
                    continue
                try:
                    outer = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(outer, dict):
                    continue
                message = outer.get("message")
                if not isinstance(message, dict):
                    continue
                if has_opencode and message.get("type") == "session.usage.updated":
                    data = message.get("data")
                    if isinstance(data, dict):
                        sid = str(data.get("sessionID") or "__default__")
                        opencode_sessions[sid] = data
                elif has_claude and message.get("_type") == "ResultMessage":
                    sid = str(message.get("session_id") or "__default__")
                    claude_results[sid] = message
    except OSError:
        return empty

    if opencode_sessions:
        total_tokens = 0
        tokens_known = True
        cost_total = 0.0
        cost_known = True
        for data in opencode_sessions.values():
            tokens = data.get("tokens")
            if not isinstance(tokens, dict):
                tokens_known = False
            else:
                cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
                fields = (
                    tokens.get("input"),
                    tokens.get("output"),
                    tokens.get("reasoning"),
                    cache.get("read"),
                    cache.get("write"),
                )
                if any(v is not None and _nonneg_int(v) is None for v in fields):
                    tokens_known = False
                else:
                    total_tokens += sum(int(v or 0) for v in fields)
            parsed_cost = _finite_nonneg_float(data.get("cost"))
            if parsed_cost is not None:
                cost_total += parsed_cost
            else:
                cost_known = False
        return {
            "total_tokens": total_tokens if tokens_known else None,
            "cost_usd": cost_total if cost_known else None,
            "currency": "USD" if cost_known else None,
            "tokens_source": "agent_raw.opencode_session_usage" if tokens_known else None,
            "cost_source": "agent_raw.opencode_session_usage" if cost_known else None,
        }

    if claude_results:
        total_tokens = 0
        tokens_known = True
        cost_total = 0.0
        cost_known = True
        for message in claude_results.values():
            parsed_cost = _finite_nonneg_float(message.get("total_cost_usd"))
            if parsed_cost is not None:
                cost_total += parsed_cost
            else:
                cost_known = False
            model_usage = message.get("model_usage")
            if isinstance(model_usage, dict) and model_usage:
                for entry in model_usage.values():
                    if not isinstance(entry, dict):
                        tokens_known = False
                        continue
                    fields = (
                        entry.get("inputTokens"),
                        entry.get("outputTokens"),
                        entry.get("cacheReadInputTokens"),
                        entry.get("cacheCreationInputTokens"),
                    )
                    if any(v is not None and _nonneg_int(v) is None for v in fields):
                        tokens_known = False
                    else:
                        total_tokens += sum(int(v or 0) for v in fields)
            else:
                tokens_known = False
        return {
            "total_tokens": total_tokens if tokens_known else None,
            "cost_usd": cost_total if cost_known else None,
            "currency": "USD" if cost_known else None,
            "tokens_source": "agent_raw.claude_result_message" if tokens_known else None,
            "cost_source": "agent_raw.claude_result_message" if cost_known else None,
        }

    return empty


def _extract_accounting_record(
    token_accounting: dict | None,
    token_usage: dict | None,
    *,
    source_prefix: str,
) -> dict:
    total_tokens = None
    tokens_source = None
    cost_value = None
    currency = None
    cost_source = None

    if isinstance(token_accounting, dict):
        tokens = token_accounting.get("tokens")
        if isinstance(tokens, dict):
            parsed_total = _nonneg_int(tokens.get("total_tokens"))
            if parsed_total is not None:
                total_tokens = parsed_total
                tokens_source = f"{source_prefix}.token_accounting"
        provider_reported = token_accounting.get("provider_reported")
        if isinstance(provider_reported, dict) and provider_reported.get("status") == "complete":
            parsed_cost = _finite_nonneg_float(provider_reported.get("amount"))
            if parsed_cost is not None:
                cost_value = parsed_cost
                raw_cur = provider_reported.get("currency")
                currency = raw_cur.strip() if isinstance(raw_cur, str) and raw_cur.strip() else "USD"
                cost_source = f"{source_prefix}.provider_reported"
        if cost_value is None:
            estimate = token_accounting.get("rate_card_estimate")
            if isinstance(estimate, dict) and estimate.get("status") == "complete":
                parsed_cost = _finite_nonneg_float(estimate.get("total"))
                if parsed_cost is not None:
                    cost_value = parsed_cost
                    raw_cur = estimate.get("currency")
                    currency = raw_cur.strip() if isinstance(raw_cur, str) and raw_cur.strip() else "USD"
                    cost_source = f"{source_prefix}.rate_card_estimate"

    if total_tokens is None and isinstance(token_usage, dict):
        total_block = token_usage.get("total")
        if isinstance(total_block, dict):
            parsed_total = _nonneg_int(total_block.get("total_tokens"))
            if parsed_total is not None:
                total_tokens = parsed_total
                tokens_source = f"{source_prefix}.token_usage"

    return {
        "total_tokens": total_tokens,
        "cost_usd": cost_value,
        "currency": currency,
        "tokens_source": tokens_source,
        "cost_source": cost_source,
    }


def resolve_run_usage(
    run_root: Path | None,
    *,
    runtime_state: dict[str, dict] | None = None,
) -> dict:
    """Resolve cumulative ``(total_tokens, cost_usd, currency)`` across OpenCode, Claude Code, and Codex runs.

    Works both while a gateway run is still live (using ``agent-raw.jsonl`` and
    ``events.sqlite`` runtime state before ``summary.json`` exists) and offline
    on sealed archives (also consulting ``summary.json``). Never sums across
    different accounting sources and never guesses missing values.
    """
    raw_usage = _usage_from_agent_raw(run_root)
    if raw_usage["total_tokens"] is not None and raw_usage["cost_usd"] is not None:
        return raw_usage

    state = runtime_state
    if state is None and run_root is not None:
        _, state = read_sqlite_snapshot(run_root)
    state = state or {}

    sqlite_usage = _extract_accounting_record(
        state.get("token_accounting"),
        state.get("token_usage"),
        source_prefix="events_sqlite",
    )

    summary = _read_json_dict(run_root / "summary.json") if run_root is not None else None
    summary_usage = (
        _extract_accounting_record(
            summary.get("token_accounting"),
            summary.get("token_usage"),
            source_prefix="summary",
        )
        if summary is not None
        else dict(total_tokens=None, cost_usd=None, currency=None, tokens_source=None, cost_source=None)
    )

    tokens = raw_usage["total_tokens"]
    tokens_source = raw_usage["tokens_source"]
    if tokens is None:
        if sqlite_usage["total_tokens"] is not None:
            tokens, tokens_source = sqlite_usage["total_tokens"], sqlite_usage["tokens_source"]
        elif summary_usage["total_tokens"] is not None:
            tokens, tokens_source = summary_usage["total_tokens"], summary_usage["tokens_source"]

    cost = raw_usage["cost_usd"]
    currency = raw_usage["currency"]
    cost_source = raw_usage["cost_source"]
    if cost is None:
        if sqlite_usage["cost_usd"] is not None:
            cost, currency, cost_source = (
                sqlite_usage["cost_usd"],
                sqlite_usage["currency"],
                sqlite_usage["cost_source"],
            )
        elif summary_usage["cost_usd"] is not None:
            cost, currency, cost_source = (
                summary_usage["cost_usd"],
                summary_usage["currency"],
                summary_usage["cost_source"],
            )

    return {
        "total_tokens": tokens,
        "cost_usd": cost,
        "currency": currency,
        "tokens_source": tokens_source,
        "cost_source": cost_source,
    }


def resolve_header_metadata(
    directory: Path | None,
    *,
    run_dir: Path | None = None,
    model_override: str | None = None,
    label_override: str | None = None,
    include_header: bool = True,
    runtime_state: dict[str, dict] | None = None,
    config_dict: dict | None = None,
) -> dict | None:
    """Resolve complete header band metadata for one recording, or ``None`` when unavailable/disabled."""
    if not include_header:
        return None
    resolved_run_root = Path(run_dir).expanduser() if run_dir is not None else find_run_root(directory)
    label_info = resolve_run_label(
        resolved_run_root,
        model_override=model_override,
        label_override=label_override,
        config_dict=config_dict,
    )
    usage_info = resolve_run_usage(resolved_run_root, runtime_state=runtime_state)
    if (
        label_info is None
        and usage_info["total_tokens"] is None
        and usage_info["cost_usd"] is None
    ):
        return None
    return {
        "label": label_info["label"] if label_info is not None else "NI-RUA",
        "backend": label_info["backend"] if label_info is not None else None,
        "model": label_info["model"] if label_info is not None else None,
        "effort": label_info["effort"] if label_info is not None else None,
        "total_tokens": usage_info["total_tokens"],
        "cost_usd": usage_info["cost_usd"],
        "currency": usage_info["currency"],
        "tokens_source": usage_info["tokens_source"],
        "cost_source": usage_info["cost_source"],
    }
