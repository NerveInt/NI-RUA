"""Present public robot evidence without exposing storage/build provenance.

The store, its integrity checks and execution identifiers remain unchanged.
Only publication copies are projected. Artifact aliases are persisted so that
model-authored programs and resumed sessions can still address exact evidence.
"""

from copy import deepcopy
import json
import os
import re

from ..contracts import canonical


_ARTIFACT = re.compile(r"artifact:[0-9a-f]{64}-[0-9a-f]{16}(?![0-9a-f])")
_ARTIFACT_ID = re.compile(r"[0-9a-f]{64}-[0-9a-f]{16}\Z")
_ALIAS = re.compile(r"artifact:ref_[A-Za-z0-9_-]+")
# A run of decimal digits is a number (counter, timestamp), not a checksum.
_HASH = re.compile(r"(?=[0-9]*[a-fA-F])[0-9a-fA-F]{16,128}\Z")
_HASH_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:0[xX])?(?=[0-9]*[a-fA-F])[0-9a-fA-F]{16,128}(?![A-Za-z0-9])")
_LABELED_HASH = re.compile(
    r"\b(?i:(?:[A-Za-z][A-Za-z0-9_-]*[_-])?"
    r"(?:sha(?:[-_ ]?\d+)?|md5|crc32|(?:git[-_ ]?)?commit|hash|digest|checksum|fingerprint))"
    r"(\s*(?:[:=_-]\s*|\s+))"
    r"([A-Za-z0-9+/_-]+={0,2})(?![A-Za-z0-9+/=_-])"
)
_HEX_DIGEST_VALUE = re.compile(r"(?:0[xX])?[0-9A-Fa-f]{7,128}\Z")
_B64_STD_VALUE = re.compile(r"[A-Za-z0-9+/]{20,128}={0,2}\Z")
_B64_URL_VALUE = re.compile(r"[A-Za-z0-9_-]{20,128}={0,2}\Z")
_LONG_ID = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z][A-Za-z0-9]*)*?)[-_]"
    r"[0-9a-f]{12,128}(?:[-_][0-9a-f]{12,128})*\b")
_CLOCK_DOMAIN = re.compile(_LONG_ID.pattern + r"/generation-[0-9]+\b")
_SHORT_ID = re.compile(r"\b(?:obs|act|job|call|plan|exp|review|clock|id)_[0-9]+\b")
_STAGE_MAPS = frozenset({
    "stages", "stage_results", "stage_plans", "checkpoint_reasons", "stage_costs", "stage_overrides",
})
_STAGE_LISTS = frozenset({"pending_stage_ids", "completed_stage_ids", "remaining_stage_ids", "checkpoint_stage_ids", "depends_on"})
_ID_PREFIX = {"observation": "obs", "obs": "obs", "action": "act", "action-plugin": "act",
    "program": "job", "job": "job", "review": "review", "task-plan": "plan", "grasp-plan": "plan",
    "preview": "plan", "batch": "plan", "plan": "plan", "shared_experience": "exp",
    **{name: "id" for name in ("snapshot", "entity", "grasp", "grasp-memory", "scene-job")}}
_HASH_FIELD = re.compile(r"(?:^|_)(?:sha(?:_?\d+)?|sh_?265|md5|crc(?:_?\d+)?|blake\d*|shasum|hash(?:es)?|digests?|checksums?|fingerprints?)(?:_|$)")
_DOCUMENT_FILE = re.compile(r"(?:^|/)[^/\s]+\.(?:urdf|usd|stl|glb|py|json|ya?ml|toml|md|pdf)(?::\d+(?:-\d+)?)?\Z", re.IGNORECASE)
_DOCUMENT_FIELDS = frozenset({
    "source_url", "source_urls", "source_link", "source_links", "source_filename",
    "source_file", "source_path", "official_sources", "repository", "repository_url",
    "repo", "repo_url", "commit", "commit_id", "commit_hash", "revision_commit", "filename",
    "documentation_url", "documentation_urls", "reference_url", "reference_urls",
    "urdf_path", "robot_model_urdf_path", "source_fingerprints",
})
#: Run-directory paths (relative to the run root) the native agent is told to use, kept verbatim through the hash
#: scrubber with every path below them: ``actor`` is its working directory; ``private/actor-source`` is the read-only
#: Harness source the instructions name when the OS sandbox is off (``actor_instructions._local_source_instructions``;
#: with the sandbox it is mounted at /harness). Every other ``<run>/private/...`` path stays scrubbed.
AGENT_DIRECTORIES = ("actor", os.path.join("private", "actor-source"))


def _alias_prefix(identifier):
    """Short-ID namespace for a matched long ID; a program call is not its job."""
    name = identifier.group(1)
    if name == "program" and identifier.group().count("-") > 1:
        return "call"
    return _ID_PREFIX.get(name, "act" if name.startswith("action-") else None)


def _document_location(value):
    if not isinstance(value, str):
        return False
    return (value.startswith(("https://", "http://", "file://", "/", "./", "../"))
            or _DOCUMENT_FILE.search(value) is not None)


def _document_collection(value):
    """Recognize source-link collections, preserving observation evidence lists."""
    if isinstance(value, str):
        return _document_location(value)
    if isinstance(value, dict):
        return bool(value) and all(_document_collection(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_document_collection(item) for item in value)
    return False


def _omit_field(key, value):
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
    if (_HASH_FIELD.search(normalized) or normalized in _DOCUMENT_FIELDS
            or normalized.startswith("repository_")
            or normalized.endswith(("_repository", "_filename", "_commit"))):
        return True
    if normalized not in {"image_url", "audio_url"} and normalized.endswith(("_url", "_urls")):
        return True
    if normalized in {"version", "calibration_id"} and isinstance(value, str) and _HASH.fullmatch(value):
        return True
    if normalized in {"source", "sources", "reference", "references", "provenance"} and _document_collection(value):
        return True
    # A URL inside source metadata is documentary; source_kind and measured
    # observation references have independent physical meaning and stay intact.
    return normalized in {"url", "link"} and _document_location(value)


def _replace_hashes(text, replacement):
    numbers = [match.span() for match in re.finditer(
        r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![A-Za-z0-9_])", text)]
    index = 0
    def replace(match):
        nonlocal index
        while index < len(numbers) and numbers[index][1] <= match.start():
            index += 1
        if index < len(numbers) and numbers[index][0] <= match.start() and match.end() <= numbers[index][1]:
            return match.group()
        return replacement(match)
    return _HASH_TOKEN.sub(replace, text)


def _is_labeled_digest_token(token: str) -> bool:
    """判断标签后的完整候选词是否为真实校验和/摘要，避免误擦蛇形错误码或普通标识符。"""
    if _HEX_DIGEST_VALUE.fullmatch(token):
        return True
    if not (22 <= len(token) <= 128):
        return False
    if _B64_STD_VALUE.fullmatch(token):
        if "+" in token or token.endswith("="):
            return True
    elif _B64_URL_VALUE.fullmatch(token):
        if token.endswith("="):
            return True
    else:
        return False
    body = token.rstrip("=")
    if not (re.search(r"[a-z]", body) and re.search(r"[A-Z]", body) and re.search(r"[0-9]", body)):
        return False
    parts = [part for part in re.split(r"[_/-]", body) if part]
    if len(parts) > 1 and (
        "/" in body
        or sum(1 for part in parts if len(part) >= 3 and (part.islower() or part.isupper())) >= 2
        or any(len(part) >= 5 and (part.islower() or part.isupper()) for part in parts)
    ):
        return False
    chunks = re.findall(r"[A-Z]?[a-z]+|[A-Z]+|\d+", body)
    return len(chunks) * 3 >= len(body)


def _has_labeled_hash(text: str) -> bool:
    return any(_is_labeled_digest_token(match.group(2)) for match in _LABELED_HASH.finditer(text))


def _strip_labeled_hashes(text: str) -> str:
    return _LABELED_HASH.sub(
        lambda match: "" if _is_labeled_digest_token(match.group(2)) else match.group(0),
        text,
    )


def _is_stage_key_map(key, item) -> bool:
    """识别以阶段 ID 为键的映射（仅豁免阶段索引版 effective_tolerances，单阶段容差字典不递归豁免）。"""
    if not isinstance(item, dict):
        return False
    if key in _STAGE_MAPS:
        return True
    return key == "effective_tolerances" and bool(item) and all(isinstance(v, dict) for v in item.values())


def metadata_findings(text):
    """Name the Harness metadata kinds still present in model-readable text; read-only.

    JSON text is decoded first, as presentation does, so escaped log text is
    judged by its actual characters.
    """
    kinds = set()

    def scan(value):
        if _ARTIFACT.search(value):
            kinds.add("raw_artifact_reference")
        if _CLOCK_DOMAIN.search(value) or any(_alias_prefix(match) for match in _LONG_ID.finditer(value)):
            kinds.add("raw_identifier")
        if _has_labeled_hash(value):
            kinds.add("labeled_checksum")
        _replace_hashes(value, lambda match: kinds.add("checksum") or match.group())

    def walk(value, *, label_keys=False):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"imageUrl", "image_url", "audioUrl", "audio_url", "data_base64"} or (
                        key == "url" and value.get("type") in {"image", "audio", "input_image", "input_audio"}):
                    continue
                if not label_keys and (_omit_field(key, item) or isinstance(key, str) and _HASH.fullmatch(key)):
                    kinds.add("provenance_field")
                if isinstance(key, str):
                    scan(key)
                walk(item, label_keys=_is_stage_key_map(key, item))
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            text_value(value)

    def text_value(value):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            walk(decoded)
            return
        for line in value.splitlines() or [value]:
            try:
                decoded = json.loads(line)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, (dict, list)):
                walk(decoded)
            else:
                scan(line)

    text_value(text)
    return sorted(kinds)


def strip_hashes(text):
    """Remove checksum prose and bare digests; keep numeric measurements exact."""
    return _replace_hashes(_strip_labeled_hashes(text), lambda match: "")


def scrub_for_fallback(value, *, root=None, label_keys=False):
    """Pure in-memory receipt/error scrubber when Store is closed or unavailable (never touches SQLite)."""
    def scrub_str(text):
        if root is not None:
            for candidate in sorted({str(root), os.path.realpath(str(root))}, key=len, reverse=True):
                if candidate and candidate in text:
                    text = re.sub(re.escape(candidate) + r"/private/program-workspaces/[^/\s\"']+/", "", text)
                    text = text.replace(candidate, "<run>")
        text = _ARTIFACT.sub("artifact:ref_unavailable", text)
        text = _CLOCK_DOMAIN.sub("clock_unavailable", text)
        text = _LONG_ID.sub(lambda match: (_alias_prefix(match) or "id") + "_unavailable", text)
        return strip_hashes(text)

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not label_keys and (_omit_field(key, item) or (isinstance(key, str) and _HASH.fullmatch(key))):
                continue
            clean_key = scrub_str(key) if isinstance(key, str) else deepcopy(key)
            if isinstance(item, str) and _HASH.fullmatch(item):
                if key in {"description", "title", "$comment"}:
                    continue
                if key == "id" or (isinstance(key, str) and key.endswith(("_id", "_ref"))):
                    result[clean_key] = "id_unavailable"
                    continue
            result[clean_key] = scrub_for_fallback(item, root=root, label_keys=_is_stage_key_map(key, item))
        return result
    if isinstance(value, list):
        return [scrub_for_fallback(item, root=root) for item in value
                if not (isinstance(item, str) and _HASH.fullmatch(item))]
    if isinstance(value, tuple):
        return tuple(scrub_for_fallback(item, root=root) for item in value)
    if isinstance(value, str):
        return scrub_str(value)
    return deepcopy(value)


class ActorPresentation:
    """Copy evidence for an actor, with stable reversible artifact addresses."""

    def __init__(self, store):
        self.store = store
        self._aliases = {}
        # 缓存短 ID 正向映射，避免每次正则扫描重复全表读取 SQLite
        self._id_aliases = {}

    def _alias(self, reference):
        # All owners sharing this Store serialize allocation through its RLock.
        # Reloading state also covers independently created presentation objects.
        with self.store.lock:
            if reference in self._aliases:
                return self._aliases[reference]
            records = self.store.states("actor_reference")
            for record in records:
                self._aliases[record["reference"]] = record["alias"]
            if reference in self._aliases:
                return self._aliases[reference]
            number = len(records) + 1
            while self.store.read_state("actor_reference", "ref_" + str(number)) is not None:
                number += 1
            identifier = "ref_" + str(number)
            alias = "artifact:" + identifier
            self.store.set_state("actor_reference", identifier,
                                 dict(alias=alias, reference=reference))
            self._aliases[reference] = alias
            return alias

    def _reference(self, alias):
        identifier = alias.removeprefix("artifact:")
        record = self.store.read_state("actor_reference", identifier)
        if (not isinstance(record, dict) or record.get("alias") != alias
                or not isinstance(record.get("reference"), str)
                or _ARTIFACT.fullmatch(record["reference"]) is None):
            raise ValueError("unknown_actor_artifact_reference")
        return record["reference"]

    def _sync_id_aliases(self):
        """在 store 锁保护下从持久化状态同步短 ID 映射表。"""
        records = self.store.states("actor_id")
        for record in records:
            self._id_aliases[record["original"]] = record["alias"]
        return records

    def _lookup_id_alias(self, value):
        """只读查询已注册的短 ID 别名，不分配新别名。"""
        with self.store.lock:
            if value in self._id_aliases:
                return self._id_aliases[value]
            self._sync_id_aliases()
            return self._id_aliases.get(value)

    def _id_alias(self, value, prefix="id"):
        with self.store.lock:
            if value in self._id_aliases:
                return self._id_aliases[value]
            records = self._sync_id_aliases()
            if value in self._id_aliases:
                return self._id_aliases[value]
            number = 1 + sum(record["alias"].startswith(prefix + "_") for record in records)
            alias = prefix + "_" + str(number)
            self.store.set_state("actor_id", alias, dict(alias=alias, original=value))
            self._id_aliases[value] = alias
            return alias

    def _log_paths(self, text):
        """Name program files relative to their workspace; run-directory paths are storage layout."""
        root = getattr(self.store, "root", None)
        if root is None:
            return text
        for candidate in sorted({str(root), os.path.realpath(str(root))}, key=len, reverse=True):
            if candidate and candidate in text:
                text = re.sub(re.escape(candidate) + r"/private/program-workspaces/[^/\s\"']+/", "", text)
                text = text.replace(candidate, "<run>")
        return text

    def present_identifiers(self, text, *, retroactive_ids=True):
        """Bind executable references to the same aliases as program replies."""
        text = _ARTIFACT.sub(lambda match: self._alias(match.group()), text)
        text = _CLOCK_DOMAIN.sub(lambda match: self._id_alias(match.group(), "clock"), text)
        def identifier(match):
            prefix = _alias_prefix(match)
            if prefix:
                return self._id_alias(match.group(), prefix)
            # 模型出口模式下禁止未知前缀裸词查询后注册的 actor_id，防止跨轮回溯改写击穿前缀缓存
            if not retroactive_ids:
                return match.group()
            existing = self._lookup_id_alias(match.group())
            return existing if existing else match.group()
        return _LONG_ID.sub(identifier, text)

    def _actor_dirs(self):
        """The run directories the native agent is told to use, as given and symlink-resolved (``AGENT_DIRECTORIES``:
        its working directory and, without the OS sandbox, the published Harness source the instructions name)."""
        root = getattr(self.store, "root", None)
        if root is None:
            return []
        paths = {os.path.join(str(root), name) for name in AGENT_DIRECTORIES}
        return sorted(paths | {os.path.realpath(path) for path in paths}, key=len, reverse=True)

    def _present_text(self, text, *, json_text=False, alias_values=False, retroactive_ids=True, log_paths=False):
        """Present text, keeping the agent's own run directories (and paths under them) verbatim.

        Both embed the run id; scrubbing it would send the native agent to a
        directory that does not exist (``AGENT_DIRECTORIES``).
        """
        kept = []
        for directory in self._actor_dirs():
            if directory in text:
                # Not a longer sibling name (actor2, actor-source.bak); a sentence-ending period may follow.
                pattern = re.escape(directory) + r"(?![\w-]|\.[\w-])(?:/[\w.-]*)*"
                def keep(match):
                    kept.append(match.group())
                    return "%ACTORDIR" + str(len(kept) - 1) + "%"
                text = re.sub(pattern, keep, text)
        text = self._present_text_scrubbed(
            text, json_text=json_text, alias_values=alias_values, retroactive_ids=retroactive_ids, log_paths=log_paths
        )
        return re.sub("%ACTORDIR([0-9]+)%", lambda match: kept[int(match.group(1))], text) if kept else text

    def _present_text_scrubbed(self, text, *, json_text=False, alias_values=False, retroactive_ids=True, log_paths=False):
        if json_text:
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                # Programs commonly print complete JSON objects among ordinary
                # log lines. Project those lines; leave other text exact.
                lines = []
                for line in text.splitlines(keepends=True):
                    try:
                        value = json.loads(line)
                    except (TypeError, ValueError):
                        lines.append(line)
                    else:
                        if isinstance(value, (dict, list)):
                            body = line.rstrip("\r\n")
                            lines.append(canonical(self._present(value, retroactive_ids=retroactive_ids)) + line[len(body):])
                        else:
                            lines.append(line)
                text = "".join(lines)
            else:
                if isinstance(value, (dict, list)):
                    return canonical(self._present(value, retroactive_ids=retroactive_ids))
        if log_paths:
            text = self._log_paths(text)
        if not alias_values:
            text = _strip_labeled_hashes(text)
        text = self.present_identifiers(text, retroactive_ids=retroactive_ids)
        if alias_values:
            text = _LONG_ID.sub(lambda match: self._id_alias(match.group()), text)
            return _replace_hashes(text, lambda match: self._id_alias(match.group()))
        text = _LONG_ID.sub(lambda match: strip_hashes(match.group()), text)
        return strip_hashes(text)

    def _present(self, value, *, log_excerpt=False, documentary=True, label_keys=False, retroactive_ids=True):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "url" and value.get("type") in {"image", "audio", "input_image", "input_audio"}:
                    result[key] = deepcopy(item)
                    continue
                if documentary and not label_keys and (_omit_field(key, item) or isinstance(key, str) and _HASH.fullmatch(key)):
                    continue
                presented_key = (self._present_text(key, alias_values=True, retroactive_ids=retroactive_ids)
                                 if isinstance(key, str) else deepcopy(key))
                json_text = documentary and (key in {"stdout_text", "stderr_text", "stdout_excerpt", "stderr_excerpt"}
                             or (key == "text" and (value.get("type") in {"inputText", "text", "input_text", "output_text"} or log_excerpt)))
                if isinstance(item, str):
                    if key in {"stdout_text", "stderr_text"}:
                        item = self._log_paths(item)
                    if key in {"imageUrl", "image_url", "audioUrl", "audio_url", "data_base64"}:
                        result[presented_key] = item
                    elif key == "artifact_id" and _ARTIFACT_ID.fullmatch(item):
                        result[presented_key] = self._alias("artifact:" + item).removeprefix("artifact:")
                    elif (key == "id" or key.endswith(("_id", "_ref"))) and (identifier := _LONG_ID.fullmatch(item)):
                        result[presented_key] = self._id_alias(item, _alias_prefix(identifier) or "id")
                    elif _HASH.fullmatch(item):
                        if key in {"description", "title", "$comment"}:
                            continue
                        if not documentary or key == "id" or key.endswith(("_id", "_ref")):
                            result[presented_key] = self._id_alias(item)
                    else:
                        result[presented_key] = self._present_text(item, json_text=json_text,
                            alias_values=not documentary and key not in {"description", "title", "$comment"},
                            retroactive_ids=retroactive_ids,
                            log_paths=documentary and key in {"stdout_text", "stderr_text", "stdout_excerpt", "stderr_excerpt", "error", "reason", "message", "detail"})
                elif key in _STAGE_LISTS and isinstance(item, list):
                    result[presented_key] = [self._present_text(label, alias_values=True, retroactive_ids=retroactive_ids)
                        if isinstance(label, str) else self._present(label, documentary=documentary, retroactive_ids=retroactive_ids)
                        for label in item]
                else:
                    result[presented_key] = self._present(item,
                        log_excerpt=key in {"stdout", "stderr", "stdout_excerpt", "stderr_excerpt"},
                        documentary=documentary, label_keys=_is_stage_key_map(key, item),
                        retroactive_ids=retroactive_ids)
            return result
        if isinstance(value, list):
            return [self._present(item, documentary=documentary, retroactive_ids=retroactive_ids) for item in value
                    if not (documentary and isinstance(item, str) and _HASH.fullmatch(item))]
        if isinstance(value, tuple):
            return tuple(self._present(item, documentary=documentary, retroactive_ids=retroactive_ids) for item in value)
        if isinstance(value, str):
            return self._present_text(value, alias_values=not documentary, retroactive_ids=retroactive_ids)
        return deepcopy(value)

    def present_data(self, value):
        """Alias typed business/display data without omitting or parsing fields."""
        return self._present(value, documentary=False)

    def present_response(self, response):
        """Render typed parts at the program boundary; retain legacy policy."""
        from .actor_output import project_metadata
        from ..tool_output import StructuredText, ToolResponse, wire_content
        if isinstance(response, ToolResponse):
            content = [StructuredText(project_metadata(item.value, item.projection))
                       if isinstance(item, StructuredText) else item for item in response.content]
            return dict(success=response.success, contentItems=wire_content(content, present=self.present))
        return self.present(response)

    def present_program_response(self, response):
        """Deliver business data separately from display; never infer it from text."""
        from .actor_output import project_metadata
        from ..tool_output import ToolResponse, validate_display, output_contract_error
        try:
            result = self.present_response(response)
        except Exception:
            result = {"success": False, "contentItems": []}
            if isinstance(response, ToolResponse) and response.contract_error is None:
                response = output_contract_error(response, None, "program_content_not_json", "/content")
        if isinstance(response, ToolResponse):
            def _safe_present(val):
                try:
                    return self.present(val)
                except Exception:
                    return scrub_for_fallback(val, root=getattr(self.store, "root", None))

            try:
                validate_display(response.data)
                data = _safe_present(project_metadata(response.data, response.data_projection))
                validate_display(data)
            except (TypeError, ValueError, OverflowError):
                # A malformed auxiliary value after execution cannot erase the receipt.
                kept, omitted = {}, []
                if isinstance(response.data, dict):
                    for key, value in response.data.items():
                        if key == 'receipt' and isinstance(value, dict):
                            receipt = {}
                            for name, field in value.items():
                                try:
                                    validate_display({name: field})
                                    receipt[name] = field
                                except (TypeError, ValueError, OverflowError):
                                    omitted.append('/data/receipt/' + str(name))
                            kept[key] = receipt
                            continue
                        try:
                            validate_display({key: value})
                            kept[key] = value
                        except (TypeError, ValueError, OverflowError):
                            omitted.append('/data/' + str(key))
                else:
                    omitted.append('/data')
                response = output_contract_error(response, None, 'program_data_not_json', '/data')
                response.contract_error['omitted_paths'] = omitted
                result['success'] = False
                data = _safe_present(project_metadata(kept, response.data_projection)) if kept else None
            result.update(kind=response.kind, data=data)
            if response.contract_error is not None:
                result['output_contract_error'] = _safe_present(response.contract_error)
        else:
            result.update(kind='untyped', data=None)
        return result

    def present_text(self, text):
        """Keep useful text and addressable IDs without checksum metadata."""
        return self._present_text(text, json_text=True)

    def present_egress_text(self, text):
        """模型出口文本清洗：完整剥离哈希与映射显式标识，禁止后注册裸词回溯改写历史前缀。"""
        return self._present_text(text, json_text=True, retroactive_ids=False)

    def present_egress(self, value):
        """模型出口结构化清洗：保持安全脱敏边界，同时锁定跨轮输出幂等性。"""
        return self._present(value, retroactive_ids=False)

    def present_egress_data(self, value):
        """模型出口 Schema 与业务数据清洗：保留字段结构且禁止后注册裸词回溯改写历史前缀。"""
        return self._present(value, documentary=False, retroactive_ids=False)

    def present(self, value):
        """Project model-visible copies; preserve measurements and raw evidence."""
        return self._present(value)

    def restore(self, value):
        """Resolve aliases in arguments, including embedded Python source text.

        This does not restore removed provenance fields. Unknown aliases fail
        before dispatch rather than becoming a guessed or redirected address.
        """
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                restored_key = self.restore(key)
                if key == "artifact_id" and isinstance(item, str) and item.startswith("ref_"):
                    result[restored_key] = self._reference("artifact:" + item).removeprefix("artifact:")
                else:
                    result[restored_key] = self.restore(item)
            return result
        if isinstance(value, list):
            return [self.restore(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.restore(item) for item in value)
        if isinstance(value, str):
            def original(match):
                record = self.store.read_state("actor_id", match.group())
                return record["original"] if record else match.group()
            return _SHORT_ID.sub(original, _ALIAS.sub(lambda match: self._reference(match.group()), value))
        return deepcopy(value)

    def present_artifact(self, info, content):
        """Return public metadata and bytes, projecting only JSON document bodies.

        Text artifacts use the same model projection. Binary artifacts retain exact bytes. Metadata omits original byte counts
        and storage paths because a projected JSON body can have another size.
        The caller must run access-scope/profile validation on the original first.
        """
        metadata = self.present(info)
        for key in ("storage_key", "byte_size"):
            metadata.pop(key, None)
        mime = str(info.get("mime_type", "")).split(";", 1)[0].strip().lower()
        if mime == "application/json" or mime.endswith("+json"):
            content = canonical(self.present(json.loads(content))).encode("utf-8")
        elif mime.startswith("text/") or mime in {"application/xml", "application/x-python", "application/javascript"}:
            content = self.present_text(content.decode("utf-8")).encode("utf-8")
        return metadata, content
