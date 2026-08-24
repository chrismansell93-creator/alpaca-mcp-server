"""
Tests for schema normalization aimed at strict function-calling clients.

Covers:
- Single-branch allOf is merged so the enum it hides becomes visible
- Multi-branch allOf is left alone
- The plural examples array becomes the singular example Gemini reads
- A type array becomes anyOf, and a numeric enum becomes a numeric range
- Author-chosen names and literal data are not mistaken for schema keywords
- Every advertised tool schema stays inside what Gemini accepts

The rules asserted here were derived by validating the real tool schemas
against ``google.genai.types.Schema``, which is generated from the same proto
the Gemini endpoint parses.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from fastmcp.client import Client

from alpaca_mcp_server.schema_compat import (
    GEMINI_SCHEMA_KEYWORDS,
    normalize_tool_schema,
)
from alpaca_mcp_server.server import build_server

DUMMY_ENV = {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_SECRET_KEY": "test-secret",
    "ALPACA_PAPER_TRADE": "true",
}

# Positions where a nested value is itself a schema, so recursion must continue.
_NAMED_MAPS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
_SUB_SCHEMAS = frozenset(
    {"items", "additionalProperties", "not", "if", "then", "else", "contains"}
)
_SCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_LITERALS = frozenset({"const", "default", "example", "examples"})


def _collect_violations(schema, path: str, found: list[str]) -> None:
    """Walk a schema and record anything Gemini's Schema proto would reject."""
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        if key not in GEMINI_SCHEMA_KEYWORDS:
            found.append(f"{path}: unsupported keyword {key!r}")
        if key == "type" and isinstance(value, list):
            found.append(f"{path}: type must be a single value, got {value!r}")
        if key == "enum" and isinstance(value, list):
            non_strings = [v for v in value if not isinstance(v, str)]
            if non_strings:
                found.append(f"{path}: enum must be strings, got {non_strings!r}")
        if key in _LITERALS:
            continue
        if key in _NAMED_MAPS and isinstance(value, dict):
            for name, entry in value.items():
                _collect_violations(entry, f"{path}.{name}", found)
        elif key in _SUB_SCHEMAS:
            _collect_violations(value, f"{path}.{key}", found)
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            for index, entry in enumerate(value):
                _collect_violations(entry, f"{path}.{key}[{index}]", found)


async def _list_tools() -> list:
    with patch.dict(os.environ, DUMMY_ENV, clear=False):
        server = build_server()
    async with Client(transport=server) as c:
        return await c.list_tools()


def test_single_branch_allof_is_merged():
    """The branch's enum and default must survive the merge."""
    schema = {
        "allOf": [
            {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "asc",
                "description": "from the branch",
            }
        ],
        "type": "string",
        "description": "from the sibling",
    }

    assert normalize_tool_schema(schema) == {
        "type": "string",
        "enum": ["asc", "desc"],
        "default": "asc",
        # The sibling is the narrower override, so it wins.
        "description": "from the sibling",
    }


def test_multi_branch_allof_is_left_alone():
    """Merging an intersection of two branches would change what it accepts."""
    schema = {"allOf": [{"type": "string"}, {"maxLength": 4}]}

    assert normalize_tool_schema(schema) == schema


def test_nested_single_branch_allof_is_merged():
    schema = {
        "type": "object",
        "properties": {
            "sort": {"allOf": [{"enum": ["asc"]}], "type": "string"},
        },
    }

    assert normalize_tool_schema(schema) == {
        "type": "object",
        "properties": {"sort": {"enum": ["asc"], "type": "string"}},
    }


def test_examples_array_becomes_singular_example():
    schema = {"type": "string", "examples": ["FILL", "TRANS"]}

    assert normalize_tool_schema(schema) == {"type": "string", "example": "FILL"}


def test_existing_example_survives_examples_removal():
    schema = {"type": "string", "example": "kept", "examples": ["dropped"]}

    assert normalize_tool_schema(schema) == {"type": "string", "example": "kept"}


def test_type_array_becomes_anyof():
    schema = {"type": ["string", "null"], "description": "kept"}

    assert normalize_tool_schema(schema) == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "kept",
    }


def test_single_entry_type_array_becomes_a_plain_type():
    schema = {"type": ["string"]}

    assert normalize_tool_schema(schema) == {"type": "string"}


def test_numeric_enum_becomes_a_range():
    schema = {"type": "integer", "enum": [0, 1, 2, 3], "description": "0=off"}

    assert normalize_tool_schema(schema) == {
        "type": "integer",
        "description": "0=off",
        "minimum": 0,
        "maximum": 3,
    }


def test_string_enum_is_kept():
    schema = {"type": "string", "enum": ["asc", "desc"]}

    assert normalize_tool_schema(schema) == schema


def test_property_named_like_a_keyword_is_not_treated_as_one():
    """Keys under properties are author-chosen names, not schema keywords."""
    schema = {
        "type": "object",
        "properties": {
            "allOf": {"type": "string"},
            "examples": {"type": "string"},
        },
    }

    assert normalize_tool_schema(schema) == schema


def test_literal_data_is_not_rewritten():
    """default holds a value, so keys inside it are data and must be untouched."""
    schema = {"type": "object", "default": {"allOf": 1, "examples": [2]}}

    assert normalize_tool_schema(schema) == schema


def test_normalization_is_idempotent():
    schema = {"allOf": [{"enum": ["asc"]}], "type": "string", "examples": ["asc"]}
    once = normalize_tool_schema(schema)

    assert normalize_tool_schema(once) == once


async def test_tool_schemas_are_gemini_compatible():
    """Guards issue #71: one rejected tool makes Gemini 400 the whole request."""
    tools = await _list_tools()

    violations: list[str] = []
    for tool in tools:
        _collect_violations(tool.inputSchema or {}, tool.name, violations)

    assert not violations, "Gemini would reject:\n" + "\n".join(sorted(violations))


async def test_sort_enum_is_visible_to_clients():
    """The enum used to be buried in an allOf that most clients never merge."""
    tools = {t.name: t.inputSchema for t in await _list_tools()}

    for name in ("get_option_bars", "get_option_trades", "get_corporate_actions"):
        sort = tools[name]["properties"]["sort"]
        assert "allOf" not in sort, f"{name}.sort still wraps allOf"
        assert sort["enum"] == ["asc", "desc"], f"{name}.sort lost its enum"
