"""Unit tests for ``ni_rua.core.schema_guide.strip_guided_schema_descriptions``."""

NI_RUA_TEST_TIER = "fast"

from copy import deepcopy
from types import SimpleNamespace
import unittest

from ni_rua.core.contracts import canonical
from ni_rua.core.schema_guide import strip_guided_schema_descriptions
from ni_rua.cortex.agents.mcp_server import mcp_tool_catalog
from ni_rua.robot.model.embodiments import ARX_X5_PROFILE, LIBERO_FRANKA_PANDA_PROFILE
from ni_rua.runtime.tools.catalog_guide import share_lines
from ni_rua.runtime.tools.definitions import public_definitions
from ni_rua.runtime.tools.plugins.robotic_view import make_robotic_view_tool


def _remaining_descriptions(node, path=""):
    found = []
    if not isinstance(node, dict):
        return found
    if isinstance(node.get("description"), str):
        found.append((path, node["description"]))
    if isinstance(node.get("properties"), dict):
        for field, child in node["properties"].items():
            found.extend(_remaining_descriptions(child, f"{path}.{field}" if path else field))
    if isinstance(node.get("items"), dict):
        found.extend(_remaining_descriptions(node["items"], f"{path}[]"))
    if isinstance(node.get("prefixItems"), list):
        for idx, child in enumerate(node["prefixItems"]):
            found.extend(_remaining_descriptions(child, f"{path}[{idx}]"))
    for key in ("oneOf", "anyOf", "allOf"):
        if isinstance(node.get(key), list):
            for child in node[key]:
                found.extend(_remaining_descriptions(child, path))
    for key in ("additionalProperties", "then", "else"):
        if isinstance(node.get(key), dict):
            found.extend(_remaining_descriptions(node[key], path))
    return found


class SchemaGuideStripTests(unittest.TestCase):
    def test_layered_v3_strips_cold_gists_and_nested_cold_fields_on_libero_and_x5(self):
        identity_pres = SimpleNamespace(present_text=lambda s: s, present_data=lambda d: deepcopy(d))
        for profile in (LIBERO_FRANKA_PANDA_PROFILE, ARX_X5_PROFILE):
            with self.subTest(robot=profile.robot_model_id):
                full_tools = public_definitions(
                    profile=profile, catalog_profile="full_v1", object_table=True, scene3d=True
                )
                compact_tools = public_definitions(
                    profile=profile, catalog_profile="compact_v1", object_table=True, scene3d=True
                )
                l1_tools = public_definitions(
                    profile=profile, catalog_profile="layered_v1", object_table=True, scene3d=True
                )
                l2_tools = share_lines(
                    public_definitions(profile=profile, catalog_profile="layered_v2", object_table=True, scene3d=True)
                )
                l3_tools = share_lines(
                    public_definitions(profile=profile, catalog_profile="layered_v3", object_table=True, scene3d=True)
                )

                full_mcp = {t["name"]: t for t in mcp_tool_catalog(full_tools, identity_pres)}
                compact_mcp = {t["name"]: t for t in mcp_tool_catalog(compact_tools, identity_pres)}
                l1_mcp = {t["name"]: t for t in mcp_tool_catalog(l1_tools, identity_pres)}
                l2_mcp = {t["name"]: t for t in mcp_tool_catalog(l2_tools, identity_pres)}
                l3_mcp = {t["name"]: t for t in mcp_tool_catalog(l3_tools, identity_pres)}

                for tool_name in ("robot_action", "geometry_query"):
                    self.assertEqual(l3_mcp[tool_name]["inputSchema"], compact_mcp[tool_name]["inputSchema"])
                    self.assertEqual(l2_mcp[tool_name]["inputSchema"], compact_mcp[tool_name]["inputSchema"])
                    self.assertEqual(l1_mcp[tool_name]["inputSchema"], compact_mcp[tool_name]["inputSchema"])
                    self.assertEqual(_remaining_descriptions(l3_mcp[tool_name]["inputSchema"]), [])
                    self.assertLess(len(canonical(l3_mcp[tool_name])), len(canonical(compact_mcp[tool_name])))
                    self.assertLess(len(canonical(l3_mcp[tool_name])), len(canonical(full_mcp[tool_name])))

                # Shared lines across tools remain stripped
                self.assertNotIn(
                    "description",
                    l3_mcp["history_view_observation"]["inputSchema"]["properties"]["focus_3d"],
                )

    def test_plugin_without_guide_and_examples_are_untouched(self):
        plugin_def = make_robotic_view_tool(lambda a, c: None, profile=ARX_X5_PROFILE).definition
        self.assertEqual(
            strip_guided_schema_descriptions(plugin_def["inputSchema"], plugin_def["description"]),
            plugin_def["inputSchema"],
        )

        # Business key named "description" inside examples/default must never be removed
        schema = {
            "type": "object",
            "properties": {
                "cold_obj": {
                    "type": "object",
                    "description": "Long cold description truncated in layered guide.",
                    "properties": {
                        "nested": {"type": "string", "description": "Nested cold description."},
                    },
                    "examples": [{"description": "keep me"}],
                },
                "warm_field": {"type": "string", "description": "Warm help text."},
                "unguided": {"type": "string", "description": "Not mentioned in guide."},
            },
        }
        description = (
            "Tool summary.\n"
            "Input field guide (use only fields applicable to the selected mode/kind).\n"
            "warm_field: string -- Warm help text.\n"
            "Rarely needed fields: type, bounds and the description, cut where it ends in \"...\". "
            "The file tool_reference.md in your working directory has one complete line per field.\n"
            "cold_obj: object -- Long cold description ..."
        )
        stripped = strip_guided_schema_descriptions(schema, description)
        self.assertNotIn("description", stripped["properties"]["cold_obj"])
        self.assertNotIn("description", stripped["properties"]["cold_obj"]["properties"]["nested"])
        self.assertEqual(stripped["properties"]["cold_obj"]["examples"], [{"description": "keep me"}])
        self.assertNotIn("description", stripped["properties"]["warm_field"])
        self.assertEqual(stripped["properties"]["unguided"]["description"], "Not mentioned in guide.")


if __name__ == "__main__":
    unittest.main()
