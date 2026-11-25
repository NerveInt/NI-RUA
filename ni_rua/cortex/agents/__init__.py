"""Backend-neutral agent runtime layer.

``base`` defines the ``AgentSession`` contract, ``AgentCapabilities``, the session errors and ``read_agent_identity``;
``factory`` selects and gates the configured backend (``create_session``, ``backend_requirements``); ``dispatch``
orders concurrent tool calls. The Codex backend lives in ``ni_rua.cortex.agents.codex.session`` (``CodexSession``).
Submodules are imported explicitly (``from ni_rua.cortex.agents.base import AgentSession``); this package imports
nothing eagerly so that loading it never pulls in a native agent SDK.
"""
