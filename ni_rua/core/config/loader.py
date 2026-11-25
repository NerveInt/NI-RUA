"""Configuration loader, Schema v2 migration, CLI metadata derivation, and launch validator."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import json
import os
from pathlib import Path
import sys
import types
from typing import Union, get_args, get_origin
from urllib.parse import urlsplit

from ..embodiment import FrozenDict
from .hooks import expand_profiles, validate_task_notes
from .sections import (
    AgentConfig,
    AnalysisConfig,
    BackendConfig,
    ConfigError,
    EmbodimentConfig,
    ExternalConfig,
    FeaturesConfig,
    ImagesConfig,
    LibraryConfig,
    ModelConfig,
    PolicyConfig,
    QueueConfig,
    RETIRED_FEATURE_FIELDS,
    RemoteConfig,
    RunConfig,
    SceneConfig,
    Section,
    _construct,
    _handoff_compat,
    _merge,
    _omit_agent_defaults,
    _section_hints,
    _section_type,
    _tuple_item_type,
    option,
)
from .thresholds import MOTION_THRESHOLD_FIELDS

LOCAL_CONFIG_PATH = Path(__file__).resolve().parents[3] / 'configs/ni-rua.local.json'
# Provider profiles directory used while agent.profiles_dir is null (git-ignored; ni_rua/cortex/agents/profiles.py).
DEFAULT_PROFILES_DIR = Path(__file__).resolve().parents[3] / 'configs/profiles'

# 多核服务器高并发（默认 60 路）下抑制各子进程默认按逻辑核数拉满线程池导致触碰 cgroup pids.max
DEFAULT_SUBPROCESS_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1',
    'OPENCV_FOR_THREADS_NUM': '1',
    'TORCH_NUM_THREADS': '2',
    'NUMBA_NUM_THREADS': '2',
    'PXR_WORK_THREAD_LIMIT': '4',
    'RAYON_NUM_THREADS': '2',
    'TOKIO_WORKER_THREADS': '2',
}

_MOTION_THRESHOLD_SET = frozenset(MOTION_THRESHOLD_FIELDS)


def migrate_v1_to_v2(doc: dict) -> dict:
    """Migrate a v1 (``"version": 1``) or partial/unversioned config mapping to Schema v2 without mutating ``doc``."""
    if not isinstance(doc, dict):
        raise ConfigError('config_object_required')
    if 'version' in doc:
        version = doc['version']
        if type(version) is not int:
            raise ConfigError('invalid_config_type:Config.version')
        if version not in (1, 2):
            raise ConfigError(f'invalid_config_choice:Config.version: {version!r}')
    out = dict(doc)
    if out.get('version') == 1:
        out['version'] = 2
    features = out.get('features')
    if isinstance(features, dict):
        moved = {}
        kept_features = {}
        for key, val in features.items():
            if key in RETIRED_FEATURE_FIELDS:
                continue
            if key in _MOTION_THRESHOLD_SET:
                moved[key] = val
            else:
                kept_features[key] = val
        if moved or len(kept_features) != len(features):
            out['features'] = kept_features
        if moved:
            raw_emb = out.get('embodiment')
            if raw_emb is None and 'embodiment' not in out:
                raw_emb = {}
            elif not isinstance(raw_emb, dict):
                raise ConfigError('config_object_required:config.embodiment')
            else:
                raw_emb = dict(raw_emb)
            existing_thr = raw_emb.get('thresholds')
            if existing_thr is None and 'thresholds' not in raw_emb:
                existing_thr = {}
            elif not isinstance(existing_thr, dict):
                raise ConfigError('invalid_config_type:EmbodimentConfig.thresholds')
            raw_emb['thresholds'] = {**moved, **existing_thr}
            out['embodiment'] = raw_emb
    return out


def _config_paths():
    """Resolve ``(core_dir, repo_root, local_config_path)`` honoring package layout and test overrides."""
    pkg = sys.modules.get('ni_rua.core.config')
    pkg_file = Path(getattr(pkg, '__file__', __file__)).resolve()
    if pkg_file.name == '__init__.py' and pkg_file.parent.name == 'config':
        core_dir = pkg_file.parents[1]
        repo_root = pkg_file.parents[3]
    else:
        core_dir = pkg_file.parent
        repo_root = pkg_file.parents[2]
    local_path = Path(getattr(pkg, 'LOCAL_CONFIG_PATH', LOCAL_CONFIG_PATH))
    return core_dir, repo_root, local_path


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f'cannot_read_config:{path}: {error}') from error
    if not isinstance(value, dict):
        raise ConfigError(f'config_object_required:{path}')
    return value


def _expand_profiles(config):
    """Expand ``source="profile"`` upstreams via registered cortex hook (no-op when none use profiles)."""
    agent = config.agent
    upstreams = (agent.claude_code.upstream, agent.opencode.upstream, agent.codex_upstream)
    if not any(upstream is not None and upstream.source == 'profile' for upstream in upstreams):
        return config
    return expand_profiles(config)


def _legacy_environment(environ):
    result = {}
    aliases = {
        'CODEX_HOME': ('model', 'codex_home'), 'NI_RUA_PRICE_CARD': ('model', 'price_card'),
        'NI_RUA_ROBODOJO_ROOT': ('backend', 'root'), 'NI_RUA_ROBODOJO_PYTHON': ('backend', 'python'),
        'NI_RUA_PI05_PYTHON': ('policy', 'python'), 'NI_RUA_PI05_CHECKPOINT': ('policy', 'checkpoint'),
        'NI_RUA_PI05_SERVER_URL': ('policy', 'server_url'),
        'NI_RUA_GRASP_MEMORY_DIR': ('features', 'grasp_memory_dir'),
    }
    _, repo_root, _ = _config_paths()
    image_path = Path(environ.get('NI_RUA_IMAGE_CONFIG', repo_root / 'configs/images.json')).expanduser()
    if environ.get('NI_RUA_IMAGE_SETTINGS'):
        try:
            result['images'] = json.loads(environ['NI_RUA_IMAGE_SETTINGS'])
        except ValueError as error:
            raise ConfigError('invalid_NI_RUA_IMAGE_SETTINGS') from error
    elif 'NI_RUA_IMAGE_CONFIG' in environ or image_path.exists():
        result['images'] = _read_json(image_path)
    for variable, (section, key) in aliases.items():
        if variable in environ:
            result.setdefault(section, {})[key] = environ[variable]
    return result


@dataclass(frozen=True)
class Config(Section):
    version: int = option(2, choices=(2,))
    run: RunConfig = field(default_factory=RunConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self):
        # Unused remote/agent sections are omitted so Codex-only manifests and snapshots stay byte-identical.
        value = super().to_dict()
        value['features'] = self.features.to_dict()
        value['embodiment'] = self.embodiment.to_dict()
        if self.remote == RemoteConfig():
            value.pop('remote', None)
        if self.agent == AgentConfig():
            value.pop('agent', None)
        else:
            _omit_agent_defaults(self.agent, value['agent'])
        return value

    def __post_init__(self):
        super().__post_init__()
        if self.remote.enabled and self.features.external_perception and self.features.vggt_height:
            endpoint = urlsplit(self.features.vggt_endpoint)
            if endpoint.hostname in ('127.0.0.1', 'localhost') and endpoint.port not in {
                    item.effective_local_port for item in self.remote.services}:
                raise ConfigError(f'remote_service_missing_for_vggt_endpoint: no remote.services local_port is {endpoint.port}')
        if not self.features.airborne_handoff and self.features.handoff_preference == 'airborne':
            raise ConfigError('airborne_handoff_preference_requires_airborne_handoff_enabled')
        if self.features.motion_execution_mode == 'feedback':
            if self.features.default_motion_speed_scale != 1:
                raise ConfigError('feedback_motion_execution_requires_default_speed_1')
            if self.features.motion_review:
                raise ConfigError('feedback_motion_execution_requires_motion_review_disabled')
            if self.features.tracking_guard:
                raise ConfigError('feedback_motion_execution_requires_tracking_guard_disabled')
        from ..observation_policy import allows_depth
        if not allows_depth(self.features.observation_profile) and any((self.features.depth_skills,
                self.features.depth_auto_feedback, self.features.depth_event_payload)):
            raise ConfigError('depth_features_require_rgbd_observation_profile')
        if self.features.depth_event_payload and not self.features.depth_auto_feedback:
            raise ConfigError('depth_event_payload_requires_depth_auto_feedback')
        if 0 < self.images.trim_max_inline_images < 24:
            raise ConfigError('trim_max_inline_images_requires_0_or_at_least_24')
        if self.features.observation_profile == 'rgb_proprio_v1' and self.features.motion_review:
            object.__setattr__(self, 'features', replace(self.features, motion_review=False))
        bound = self.features.__dict__.get('_embodiment_thresholds')
        if bound is not None and bound != self.embodiment.thresholds:
            object.__setattr__(self, 'features', replace(self.features))
        object.__setattr__(self.features, '_embodiment_thresholds', self.embodiment.thresholds)

    @classmethod
    def from_dict(cls, value, *, base=None):
        """Read an archived/full or partial document without consulting environment."""
        migrated = migrate_v1_to_v2(value)
        defaults = {**cls().to_dict(), 'agent': AgentConfig().to_dict()}
        return _construct(cls, _merge(defaults, _handoff_compat(migrated)), base=Path(base) if base else None)

    @classmethod
    def load(cls, path=None, *, environ=None, overrides=None, ignore_local: bool = False):
        """Load configuration with precedence: defaults < env < ni-rua.json < ni-rua.local.json < --config < CLI.

        ``ignore_local`` (bool): when True, skips ``ni-rua.local.json`` for hermetic runs.
        """
        environ = dict(os.environ if environ is None else environ)
        value = _merge(cls().to_dict(), _legacy_environment(environ))
        value = cls.from_dict(value).to_dict()
        selected = path or environ.get('NI_RUA_CONFIG')
        core_dir, repo_root, local_path = _config_paths()
        default = repo_root / 'configs/ni-rua.json'
        if not default.exists():
            default = core_dir / 'default_config.json'
        sources = [default] if default.exists() else []
        if not ignore_local and local_path.exists():
            sources.append(local_path)
        if selected:
            sources.append(Path(selected))
        for source in sources:
            source = source.expanduser().resolve()
            document = migrate_v1_to_v2(_read_json(source))

            def normalize(data, schema):
                if not isinstance(data, dict):
                    raise ConfigError('config_object_required')
                declared = {item.name: item for item in fields(schema)}
                hints = _section_hints(schema)
                out = {}
                for key, part in data.items():
                    if schema is FeaturesConfig and key in RETIRED_FEATURE_FIELDS:
                        continue
                    if key not in declared:
                        raise ConfigError(f'unknown_config_keys:{schema.__name__}.{key}')
                    item = declared[key]
                    section = _section_type(hints[key])
                    tuple_inner = _tuple_item_type(hints[key])
                    if section is not None and (part is not None or is_dataclass(hints[key])):
                        part = normalize(part, section)
                    elif tuple_inner is not None and isinstance(tuple_inner, type) and is_dataclass(tuple_inner) and isinstance(part, list):
                        part = [normalize(entry, tuple_inner) for entry in part]
                    elif item.metadata.get('path') and isinstance(part, str):
                        part = str((source.parent / Path(part).expanduser()).resolve()) if part.strip() else part
                    out[key] = part
                return out

            normalized = _handoff_compat(normalize(document, cls))
            cls.from_dict(normalized)
            value = _merge(value, normalized)
        if overrides:
            value = _merge(value, _handoff_compat(migrate_v1_to_v2(overrides)))
        return _expand_profiles(cls.from_dict(value))

    def with_overrides(self, overrides):
        return type(self).from_dict(_merge(self.to_dict(), _handoff_compat(migrate_v1_to_v2(overrides))))

    def validate_launch(self):
        run = self.run
        if run.benchmark and self.features.motion_execution_mode != 'resampled':
            raise ConfigError('benchmark_requires_resampled_unified_action; feedback_is_diagnostic_only')
        if not run.replay and 'task_plan' in self.features.disabled_tools:
            raise ConfigError('required_task_plan_tool_cannot_be_disabled')
        if run.resume and run.replay:
            raise ConfigError('--resume and --replay are mutually exclusive')
        if (run.resume or run.replay) and not run.run_dir:
            raise ConfigError('--resume/--replay requires --run-dir')
        if run.benchmark and (run.max_actions is not None or run.wall_budget_s is not None):
            raise ConfigError('--benchmark uses official native step and model-call limits; remove --max-actions and --wall-budget-s')
        if run.benchmark and (run.resume or run.replay or run.prompt):
            raise ConfigError('--benchmark requires a fresh run and uses the native instruction, not --prompt')
        if self.features.completion_guidance in {'initial', 'both'} and not run.benchmark and not (run.resume or run.replay):
            raise ConfigError('--completion-guidance requires --benchmark')
        if run.task_notes is not None or run.task_notes_source is not None:
            if not run.benchmark and self.agent.backend != 'external':
                raise ConfigError('--task-notes requires --benchmark or --agent-backend external')
            validate_task_notes(run.task_notes, run.task_notes_source)
        return self


@dataclass(frozen=True)
class CliOptionSpec:
    """Single-source declarative CLI/queue option metadata derived from a ``Section`` field."""

    dest: str
    flag: str
    section: str
    field_name: str
    kind: str  # 'bool', 'int', 'float', 'str', 'tag', 'str_list', 'nullable_choice', 'support_z'
    choices: tuple | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    path: bool = False
    help: str | None = None
    queue_case: bool = False
    queue_required: bool = False


def _classify_cli_kind(section_name: str, field_name: str, annotation, choices) -> str:
    if section_name == 'run' and field_name == 'tag':
        return 'tag'
    if section_name == 'features' and field_name == 'vggt_support_z_m':
        return 'support_z'
    if _tuple_item_type(annotation) is str:
        return 'str_list'
    origin = get_origin(annotation)
    args = get_args(annotation) if origin in (Union, types.UnionType) else (annotation,)
    allows_none = type(None) in args
    non_none = tuple(arg for arg in args if arg is not type(None))
    if non_none == (bool,):
        return 'bool'
    if non_none == (int,):
        return 'int'
    if non_none == (float,):
        return 'float'
    if non_none == (str,) and allows_none and choices is not None:
        return 'nullable_choice'
    if non_none == (str,):
        return 'str'
    raise TypeError(f'unsupported_cli_option_type:{section_name}.{field_name}:{annotation!r}')


def _derive_cli_options(section_pairs, *, meta_key: str) -> tuple[CliOptionSpec, ...]:
    specs: list[CliOptionSpec] = []
    for section_name, section_cls in section_pairs:
        hints = _section_hints(section_cls)
        for item in fields(section_cls):
            meta = item.metadata
            cli_val = meta.get(meta_key, False)
            if not cli_val:
                continue
            dest = item.name if cli_val is True else str(cli_val)
            choices = tuple(meta['choices']) if meta.get('choices') is not None else None
            specs.append(
                CliOptionSpec(
                    dest=dest,
                    flag='--' + dest.replace('_', '-'),
                    section=section_name,
                    field_name=item.name,
                    kind=_classify_cli_kind(section_name, item.name, hints[item.name], choices),
                    choices=choices,
                    minimum=meta.get('minimum'),
                    maximum=meta.get('maximum'),
                    path=bool(meta.get('path', False)),
                    help=meta.get('cli_help'),
                    queue_case=bool(meta.get('queue_case', False)),
                    queue_required=bool(meta.get('queue_required', False)),
                )
            )
    return tuple(specs)


CLI_OPTIONS = _derive_cli_options(
    (
        ('run', RunConfig),
        ('features', FeaturesConfig),
        ('model', ModelConfig),
        ('images', ImagesConfig),
        ('remote', RemoteConfig),
        ('agent', AgentConfig),
        ('agent.external', ExternalConfig),
    ),
    meta_key='cli',
)
CLI_OPTIONS_BY_DEST = FrozenDict((spec.dest, spec) for spec in CLI_OPTIONS)
QUEUE_CASE_OPTIONS = tuple(spec for spec in CLI_OPTIONS if spec.queue_case)
QUEUE_CLI_OPTIONS = _derive_cli_options((('queue', QueueConfig),), meta_key='queue_cli')

RUN_FIELDS = frozenset(item.name for item in fields(RunConfig))
FEATURE_FIELDS = frozenset(item.name for item in fields(FeaturesConfig))
CLI_FIELDS = {
    **{name: ('run', name) for name in RUN_FIELDS},
    **{name: ('features', name) for name in FEATURE_FIELDS},
    **{spec.dest: (spec.section, spec.field_name) for spec in CLI_OPTIONS},
}


def cli_overrides(values):
    result = {}
    for name, value in values.items():
        if name in CLI_FIELDS:
            section, key = CLI_FIELDS[name]
            target = result
            for part in section.split('.'):
                target = target.setdefault(part, {})
            target[key] = value
    return result


def namespace_values(config):
    def _read_section(obj, section):
        for part in section.split('.'):
            obj = getattr(obj, part)
        return obj

    return {name: getattr(_read_section(config, section), key) for name, (section, key) in CLI_FIELDS.items()}
