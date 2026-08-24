"""
Schema compatibility for strict function-calling APIs.

Gemini's function-calling API validates tool schemas against a fixed subset of
JSON Schema and returns HTTP 400 for anything outside it, which kills every
chat turn that includes an affected tool. Four things FastMCP emits fall
outside that subset, and this module rewrites them when tools are listed:

- ``allOf``, which Gemini has no equivalent for
- the plural ``examples`` array, where Gemini reads a singular ``example``
- a ``type`` array such as ``["string", "null"]``, where Gemini takes one type
- an ``enum`` of numbers, where Gemini requires strings

These apply for every client rather than sitting behind a compatibility flag.
The ``allOf`` and ``type`` rewrites are exact. The other two keep less detail —
the first example rather than all of them, and a numeric range rather than the
exact values — which the parameter descriptions already spell out.

Flattening ``allOf`` also fixes a real bug: Alpaca's specs wrap enum parameters
in a single-branch ``allOf``, which hides the allowed values from any client
that does not merge it.

This runs at ``tools/list`` time because that is the only point where schemas
are fully resolved. Earlier hooks still contain ``$ref`` pointers into a
``$defs`` block that FastMCP inlines afterwards.
"""

from __future__ import annotations

from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool

# Accepted by Gemini's Schema proto. Sourced from google.genai.types.Schema.
GEMINI_SCHEMA_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "anyOf",
        "default",
        "description",
        "enum",
        "example",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "nullable",
        "pattern",
        "properties",
        "propertyOrdering",
        "required",
        "title",
        "type",
        "$defs",
        "$ref",
    }
)

# Keys whose values are arbitrary data rather than nested schemas.
_LITERAL_FIELDS = frozenset({"const", "default", "example", "examples"})

# Keys whose child keys are author-chosen names rather than schema keywords.
_NAMED_MAP_FIELDS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


def _merge_single_branch_allof(node: dict[str, Any]) -> dict[str, Any]:
    """Merge a one-branch ``allOf`` into its parent.

    A single branch is an identity, so merging cannot change what the schema
    accepts. Sibling keys win on conflict because they are the narrower
    override.
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


def _singularize_examples(node: dict[str, Any]) -> dict[str, Any]:
    """Replace an ``examples`` array with the singular ``example`` Gemini reads."""
    examples = node.get("examples")
    if not isinstance(examples, list) or not examples:
        return node

    normalized = {key: value for key, value in node.items() if key != "examples"}
    normalized.setdefault("example", examples[0])
    return normalized


def _split_type_union(node: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a ``type`` array as ``anyOf``, which Gemini accepts.

    One branch per type is exactly equivalent, and it is already the form
    FastMCP produces for other nullable parameters.
    """
    types = node.get("type")
    if not isinstance(types, list) or not types or "anyOf" in node:
        return node

    rest = {key: value for key, value in node.items() if key != "type"}
    if len(types) == 1:
        return {"type": types[0], **rest}
    return {"anyOf": [{"type": entry} for entry in types], **rest}


def _drop_non_string_enum(node: dict[str, Any]) -> dict[str, Any]:
    """Drop an ``enum`` of numbers, keeping its range as ``minimum``/``maximum``.

    Gemini's ``enum`` holds strings only. Stringifying the values would invite
    a client to send ``"2"`` where the API needs ``2``, so the bounds are kept
    instead. Descriptions for these parameters already spell out each value.
    """
    values = node.get("enum")
    if not isinstance(values, list) or not values:
        return node
    if all(isinstance(value, str) for value in values):
        return node

    normalized = {key: value for key, value in node.items() if key != "enum"}
    numbers = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numbers) == len(values):
        normalized.setdefault("minimum", min(numbers))
        normalized.setdefault("maximum", max(numbers))
    return normalized


def normalize_tool_schema(value: Any) -> Any:
    """Rewrite schema keywords that strict function-calling APIs reject."""
    if isinstance(value, list):
        return [normalize_tool_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    node: dict[str, Any] = {}
    for key, item in value.items():
        if key in _LITERAL_FIELDS:
            node[key] = item
        elif key in _NAMED_MAP_FIELDS and isinstance(item, dict):
            node[key] = {
                name: normalize_tool_schema(entry) for name, entry in item.items()
            }
        else:
            node[key] = normalize_tool_schema(item)

    node = _merge_single_branch_allof(node)
    node = _split_type_union(node)
    node = _drop_non_string_enum(node)
    return _singularize_examples(node)


class SchemaCompatMiddleware(Middleware):
    """Normalizes advertised tool schemas for strict function-calling clients."""

    async def on_list_tools(
        self, context: MiddlewareContext, call_next
    ) -> list[Tool]:
        tools = await call_next(context)
        return [
            tool.model_copy(update={"parameters": normalize_tool_schema(tool.parameters)})
            if isinstance(tool.parameters, dict)
            else tool
            for tool in tools
        ]
