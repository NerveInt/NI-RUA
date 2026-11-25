"""Command-line entry point for one workbench run."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from types import SimpleNamespace

from ...core.config import (
    CLI_OPTIONS,
    CliOptionSpec,
    Config,
    ConfigError,
    DEFAULT_SUBPROCESS_THREAD_ENV,
    cli_overrides,
    namespace_values,
)

for _thread_key, _thread_val in DEFAULT_SUBPROCESS_THREAD_ENV.items():
    os.environ.setdefault(_thread_key, _thread_val)

from ...runtime.tools.builtins import CONFIG_TOOL_NAMES, create_registry
from .server import serve
from .workbench import _configured_tool_specs


def _parse_vggt_support_z_m(value):
    return None if value == 'observation' else value if value == 'profile' else float(value)


def _add_cli_option(parser: argparse.ArgumentParser, spec: CliOptionSpec) -> None:
    help_kw = {'help': spec.help} if spec.help else {}
    if spec.kind == 'bool':
        parser.add_argument(spec.flag, action=argparse.BooleanOptionalAction, **help_kw)
    elif spec.kind == 'tag':
        parser.add_argument(spec.flag, action='append', **help_kw)
        parser.add_argument(
            '--clear-tags',
            action='store_const',
            const=[],
            dest=spec.dest,
            help='Clear tags inherited from the config file',
        )
    elif spec.kind == 'str_list':
        parser.add_argument(spec.flag, nargs='*', **help_kw)
    elif spec.kind == 'nullable_choice':
        parser.add_argument(spec.flag, choices=list(spec.choices or ()), **help_kw)
        parser.add_argument(
            '--no-' + spec.flag[2:],
            action='store_const',
            const=None,
            dest=spec.dest,
            help='Disable the optional visual helper',
        )
    elif spec.kind == 'support_z':
        parser.add_argument(spec.flag, type=_parse_vggt_support_z_m, **help_kw)
    else:
        kwargs: dict = dict(help_kw)
        if spec.kind == 'int':
            kwargs['type'] = int
        elif spec.kind == 'float':
            kwargs['type'] = float
        if spec.choices is not None:
            kwargs['choices'] = spec.choices
        parser.add_argument(spec.flag, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    # Single-source CLI_OPTIONS derived from Config sections (e.g. --agent-backend choices=('codex', 'claude_code', 'opencode', 'external')).
    # Literal parser.add_argument('--config', ...) and Config.load(...) are preserved for queue.require_config_gateway() AST checks.
    parser = argparse.ArgumentParser(
        description='Run NI-RUA + RoboDojo with a persistent native agent workbench',
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument('--config', help='Unified JSON config; defaults to NI_RUA_CONFIG or configs/ni-rua.json')
    parser.add_argument(
        '--no-local-config',
        action='store_true',
        help='Ignore configs/ni-rua.local.json when resolving configuration (hermetic mode)',
    )
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print resolved configuration and exit without starting services',
    )
    for spec in CLI_OPTIONS:
        _add_cli_option(parser, spec)
    return parser


def main():
    parser = build_parser()
    parsed = vars(parser.parse_args())
    try:
        config = Config.load(
            parsed.get('config'),
            overrides=cli_overrides(parsed),
            ignore_local=bool(parsed.get('no_local_config', False)),
        ).validate_launch()
        catalog_runtime = SimpleNamespace(config=config, features=config.features, embodiment=None)
        unknown_tools = (
            set(config.features.disabled_tools)
            - create_registry(_configured_tool_specs(config), runtime=catalog_runtime).names
            - CONFIG_TOOL_NAMES
        )
        if unknown_tools:
            raise ConfigError('unknown_disabled_tools:' + ','.join(sorted(unknown_tools)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if parsed.get('print_config'):
        print(json.dumps(config.to_dict(), indent=2))
        return
    args = argparse.Namespace(**namespace_values(config), resolved_config=config)
    asyncio.run(serve(args))


if __name__ == '__main__':
    main()
