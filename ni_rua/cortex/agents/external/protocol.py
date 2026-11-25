"""Names the external agent backend shares with the agent plugin and the tool registry (no imports, cheap to load).

``BACKEND_KIND``: ``agent.backend`` / ``AgentSession.backend_kind`` value. ``RUN_BEGIN`` / ``RUN_FINISH``: the two
session-bracket tools (``ni_rua/runtime/tools/plugins/run_bracket.py``). ``MCP_SERVER_NAME``: ``serverInfo.name`` of the
endpoint ``http://127.0.0.1:<agent.external.mcp_port>/mcp`` (``mcp_server.MCP_PATH``), the name host plugins register.
``SESSION_EFFECT``: the declared ``ToolSpec.effect`` of a call that changes only the agent-session bracket (no
observation, plan, memory or physical effect); the external transport reads it through ``ToolRegistry.effect_of``.
"""
BACKEND_KIND = "external"
RUN_BEGIN, RUN_FINISH = "run_begin", "run_finish"
MCP_SERVER_NAME = "ni_rua"
SESSION_EFFECT = "session"
# ni-rua-plugin doctor 探测客户端名，用于避免覆盖已附着真实 Agent 的展示信息
DOCTOR_CLIENT_NAME = "ni-rua-plugin-doctor"
