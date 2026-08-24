"""
Flatten one-branch ``allOf`` wrappers in advertised tool schemas.

FastMCP leaves query-parameter enums wrapped as
``{"allOf": [{"$ref": "..."}]}``. After ``$ref`` is inlined, clients that do
not merge ``allOf`` never see the ``enum`` — so the model has to guess
``asc``/``desc`` on sort parameters. A one-branch ``allOf`` is an identity,
so flattening it does not change what the schema accepts.

This runs at ``tools/list`` time because that is the only point where schemas
are fully resolved. Earlier hooks still contain ``$ref`` pointers into a
``$defs`` block that FastMCP inlines afterwards.
"""

from __future__ import annotations

from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool

# Keys whose values are arbitrary data rather than nested schemas.
_LITERAL_FIELDS = frozenset({"const", "default", "example", "examples"})

# Keys whose child keys are author-chosen names rather than schema keywords.
_NAMED_MAP_FIELDS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


def _merge_single_branch_allof(node: dict[str, Any]) -> dict[str, Any]:
    """Merge a one-branch ``allOf`` into its parent.

    Sibling keys win on conflict because they are the narrower override.
    Multi-branch ``allOf`` is left alone — merging it would change the
    intersection.
    """
    branches = node.get("allOf")
    if not isinstance(branches, list) or len(branches) != 1:
        return node
    if not isinstance(branches[0], dict):
        return node

    merged = dict(branches[0])
    for key, value in node.items():
        if key != "allOf":
            merged[key] = value
    return merged


def flatten_single_branch_allof(value: Any) -> Any:
    """Walk a schema and flatten every one-branch ``allOf``."""
    if isinstance(value, list):
        return [flatten_single_branch_allof(item) for item in value]
    if not isinstance(value, dict):
        return value

    node: dict[str, Any] = {}
    for key, item in value.items():
        if key in _LITERAL_FIELDS:
            node[key] = item
        elif key in _NAMED_MAP_FIELDS and isinstance(item, dict):
            node[key] = {
                name: flatten_single_branch_allof(entry) for name, entry in item.items()
            }
        else:
            node[key] = flatten_single_branch_allof(item)

    return _merge_single_branch_allof(node)


class SchemaCompatMiddleware(Middleware):
    """Flattens one-branch ``allOf`` on advertised tool schemas."""

    async def on_list_tools(
        self, context: MiddlewareContext, call_next
    ) -> list[Tool]:
        tools = await call_next(context)
        return [
            tool.model_copy(
                update={"parameters": flatten_single_branch_allof(tool.parameters)}
            )
            if isinstance(tool.parameters, dict)
            else tool
            for tool in tools
        ]
