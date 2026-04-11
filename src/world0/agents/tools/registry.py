"""Tool registry — typed, permission-gated tool definitions.

Inspired by claw-code's GlobalToolRegistry. Each tool declares:
- name, description, parameter schema
- a callable that executes the tool
- permission level required

The registry generates LLM-compatible tool specs (OpenAI function-calling
format) and dispatches tool calls by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Permission(str, Enum):
    """Permission levels for tool access."""
    READ = "read"           # Read-only queries
    WRITE = "write"         # Modify the concept world
    ADMIN = "admin"         # Destructive operations (prune, reset)


@dataclass
class ToolParam:
    """A single parameter for a tool."""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        return self.output


@dataclass
class Tool:
    """A registered tool that the LLM can invoke."""
    name: str
    description: str
    parameters: list[ToolParam]
    handler: Callable[..., ToolResult]
    permission: Permission = Permission.READ

    def to_openai_spec(self) -> dict:
        """Generate OpenAI function-calling compatible schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_spec(self) -> dict:
        """Generate Anthropic tool-use compatible schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        return self.handler(**kwargs)


class ToolRegistry:
    """Central registry for all PKM Agent tools.

    Provides:
    - Tool registration and lookup
    - LLM-compatible spec generation (OpenAI + Anthropic formats)
    - Permission-gated dispatch
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_specs(self, max_permission: Permission = Permission.ADMIN) -> list[dict]:
        """Generate OpenAI function-calling specs for allowed tools."""
        perm_order = [Permission.READ, Permission.WRITE, Permission.ADMIN]
        max_idx = perm_order.index(max_permission)
        return [
            t.to_openai_spec()
            for t in self._tools.values()
            if perm_order.index(t.permission) <= max_idx
        ]

    def anthropic_specs(self, max_permission: Permission = Permission.ADMIN) -> list[dict]:
        """Generate Anthropic tool-use specs for allowed tools."""
        perm_order = [Permission.READ, Permission.WRITE, Permission.ADMIN]
        max_idx = perm_order.index(max_permission)
        return [
            t.to_anthropic_spec()
            for t in self._tools.values()
            if perm_order.index(t.permission) <= max_idx
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        max_permission: Permission = Permission.ADMIN,
    ) -> ToolResult:
        """Dispatch a tool call by name with permission checking."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {name}. Available: {', '.join(self.names())}",
            )

        perm_order = [Permission.READ, Permission.WRITE, Permission.ADMIN]
        if perm_order.index(tool.permission) > perm_order.index(max_permission):
            return ToolResult(
                success=False,
                output=f"Permission denied: {name} requires '{tool.permission.value}' "
                       f"but current level is '{max_permission.value}'.",
            )

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(success=False, output=f"Tool error [{name}]: {e}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
