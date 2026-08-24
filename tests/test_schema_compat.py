"""
Tests for flattening one-branch allOf wrappers in tool schemas.

Covers:
- Single-branch allOf is merged so the enum it hides becomes visible
- Multi-branch allOf is left alone
- Author-chosen names and literal data are not mistaken for schema keywords
- Advertised sort parameters expose their enum to clients
"""

from __future__ import annotations

import os
from unittest.mock import patch

from fastmcp.client import Client

from alpaca_mcp_server.schema_compat import flatten_single_branch_allof
from alpaca_mcp_server.server import build_server

DUMMY_ENV = {
    "ALPACA_API_KEY": "test-key",
    "ALPACA_SECRET_KEY": "test-secret",
    "ALPACA_PAPER_TRADE": "true",
}


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

    assert flatten_single_branch_allof(schema) == {
        "type": "string",
        "enum": ["asc", "desc"],
        "default": "asc",
        # The sibling is the narrower override, so it wins.
        "description": "from the sibling",
    }


def test_multi_branch_allof_is_left_alone():
    """Merging an intersection of two branches would change what it accepts."""
    schema = {"allOf": [{"type": "string"}, {"maxLength": 4}]}

    assert flatten_single_branch_allof(schema) == schema


def test_nested_single_branch_allof_is_merged():
    schema = {
        "type": "object",
        "properties": {
            "sort": {"allOf": [{"enum": ["asc"]}], "type": "string"},
        },
    }

    assert flatten_single_branch_allof(schema) == {
        "type": "object",
        "properties": {"sort": {"enum": ["asc"], "type": "string"}},
    }


def test_property_named_like_a_keyword_is_not_treated_as_one():
    """Keys under properties are author-chosen names, not schema keywords."""
    schema = {
        "type": "object",
        "properties": {
            "allOf": {"type": "string"},
        },
    }

    assert flatten_single_branch_allof(schema) == schema


def test_literal_data_is_not_rewritten():
    """default holds a value, so keys inside it are data and must be untouched."""
    schema = {"type": "object", "default": {"allOf": [{"enum": ["x"]}]}}

    assert flatten_single_branch_allof(schema) == schema


def test_flattening_is_idempotent():
    schema = {"allOf": [{"enum": ["asc"]}], "type": "string"}
    once = flatten_single_branch_allof(schema)

    assert flatten_single_branch_allof(once) == once


async def test_sort_enum_is_visible_to_clients():
    """The enum used to be buried in an allOf that most clients never merge."""
    tools = {t.name: t.inputSchema for t in await _list_tools()}

    for name in ("get_option_bars", "get_option_trades", "get_corporate_actions"):
        sort = tools[name]["properties"]["sort"]
        assert "allOf" not in sort, f"{name}.sort still wraps allOf"
        assert sort["enum"] == ["asc", "desc"], f"{name}.sort lost its enum"
