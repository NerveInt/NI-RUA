"""Re-export of ``ni_rua.runtime.tools.tool_search`` for optional plugin discovery."""
from ..tool_search import (  # noqa: F401
    TOOL_SEARCH_DESCRIPTION,
    TOOL_SEARCH_INPUT_SCHEMA,
    TOOL_SEARCH_NAME,
    TOOL_SEARCH_OUTPUT_SCHEMA,
    UNLOCKED_DEFERRED_TOOLS_SLOT,
    handle_tool_search,
    make_tool_search_tool,
    strip_mcp_tool_prefix,
)
