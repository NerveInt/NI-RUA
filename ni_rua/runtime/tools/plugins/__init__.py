"""Optional tools that build their own ToolSpec."""

from __future__ import annotations


def configured_tool_specs(config, tool_specs=(), owner=None):
    """Load optional plugin tools enabled by ``config``; bind runtime handlers to ``owner``.

    ``agent.backend == 'external'`` appends the session-bracket tools ``run_begin`` and ``run_finish``
    (``plugins/run_bracket.py``), bound to the owner's session and Harness action budget.
    """
    backend = getattr(getattr(config, "agent", None), "backend", "codex")
    if backend == "external":
        from .run_bracket import make_run_bracket_tools

        def begin(correlation):
            if owner is None:
                raise RuntimeError("run_bracket_owner_not_bound")
            return owner.session.run_begin(correlation, budget=owner.harness.action_budget())

        def finish(summary, correlation):
            if owner is None:
                raise RuntimeError("run_bracket_owner_not_bound")
            return owner.session.run_finish(summary, correlation)

        tool_specs = (*tool_specs, *make_run_bracket_tools(begin, finish))
    if config.features.robotic_view:
        from .robotic_view import make_robotic_view_tool, read_robotic_view

        def read_view(arguments, correlation):
            if owner is None:
                raise RuntimeError("robotic_view_owner_not_bound")
            return read_robotic_view(owner.harness, arguments, correlation)

        owner_profile = getattr(getattr(owner, "harness", None), "embodiment", None)
        tool_specs = (*tool_specs, make_robotic_view_tool(read_view, profile=owner_profile))
    if config.features.stereo_guide:
        from .stereo_guide import make_stereo_guide_tool, read_stereo_guide

        def read_stereo(arguments, correlation):
            if owner is None:
                raise RuntimeError("stereo_guide_owner_not_bound")
            return read_stereo_guide(owner.harness, arguments, correlation)

        tool_specs = (*tool_specs, make_stereo_guide_tool(read_stereo))
    agent_cfg = getattr(config, "agent", None)
    want_tool_search = (
        (backend == "opencode" and bool(getattr(getattr(agent_cfg, "opencode", None), "tool_search", False)))
        or (backend == "claude_code" and bool(getattr(getattr(agent_cfg, "claude_code", None), "tool_search", False)))
    )
    if want_tool_search and not any(getattr(spec, "name", None) == "tool_search" for spec in tool_specs):
        from .tool_search import make_tool_search_tool

        tool_specs = (*tool_specs, make_tool_search_tool())
    return tool_specs
