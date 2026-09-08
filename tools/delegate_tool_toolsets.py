"""Child toolset resolution for delegate_task: what a child may and may never use."""

from __future__ import annotations

import logging
from typing import List, Optional

from toolsets import TOOLSETS
from tools.delegate_tool_config import _get_configured_child_toolsets, _get_inherit_mcp_toolsets

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "cronjob_manage",  # no scheduling more work in the parent's name
    ]
)
DEFAULT_TOOLSETS = ["terminal", "file", "web"]

def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry
        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))

def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Add every toolset whose tools are a subset of the parent's tools: a parent on a composite like ``hermes-cli``
    must still let a child request ``web``/``terminal``; bare name intersection would reject them."""
    parent_tool_names = {t for ts_name in parent_toolsets for t in (TOOLSETS.get(ts_name) or {}).get("tools", [])}
    expanded = set(parent_toolsets)
    if parent_tool_names:
        expanded.update(
            ts_name for ts_name, ts_def in TOOLSETS.items()
            if ts_name not in expanded and ts_def.get("tools") and set(ts_def["tools"]).issubset(parent_tool_names)
        )
    return expanded


def _bound_room_policy_toolsets() -> Optional[set]:
    """Current target-issued RoomLink authority ceiling, if this is a hosted turn.

    ``delegation.child_toolsets`` may intentionally exceed ordinary parent
    visibility, but it cannot exceed an execution policy that was authenticated
    and bound to this hosted turn.
    """
    try:
        from gateway.hosted_room_execution_policy import current_room_execution_policy
        policy = current_room_execution_policy()
    except Exception:
        logger.warning("Could not read bound room execution policy; denying child tools", exc_info=True)
        return set()
    if policy is None:
        return None
    enabled = getattr(policy, "enabled_toolsets", ())
    if not isinstance(enabled, (list, tuple, set, frozenset)):
        logger.warning("Bound room execution policy has invalid toolsets; denying child tools")
        return set()
    return _expand_parent_toolsets(set(enabled))


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets whose tools are ALL blocked (derived from DELEGATE_BLOCKED_TOOLS so the two can't drift) plus
    composite toolsets children must never get (``delegation``, ``kanban``)."""
    blocked_toolset_names = {"delegation", "kanban"} | {
        name for name, defn in TOOLSETS.items() if all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    return [t for t in toolsets if t not in blocked_toolset_names]

def _blocked_toolsets_for_role(role: str) -> List[str]:
    """One-tool deny toolsets for the role; passed as ``disabled_toolsets`` so
    blocked names inside mixed bundles are subtracted AFTER composite expansion."""
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    if role == "orchestrator":
        blocked_names.discard("delegate_task")
    return sorted(
        name for name, defn in TOOLSETS.items() if defn.get("tools") and set(defn.get("tools", ())).issubset(blocked_names)
    )


def _tool_names(toolsets: List[str]) -> set[str]:
    """Resolve a list of known toolsets to its effective tool names."""
    from toolsets import resolve_toolset

    return {tool for toolset in toolsets for tool in resolve_toolset(toolset)}


def _configured_child_toolsets_allow(toolsets: List[str]) -> Optional[bool]:
    """Whether a configured worker policy allows a trusted caller's narrowing.

    ``None`` means no configured policy, so the caller must use legacy
    parent-surface rules. ``False`` is intentionally fail-closed.
    """
    configured = _get_configured_child_toolsets()
    if configured is None:
        return None
    from toolsets import validate_toolset

    if not all(isinstance(toolset, str) and validate_toolset(toolset) for toolset in toolsets):
        return False
    try:
        return _tool_names(toolsets).issubset(_tool_names(configured))
    except Exception:
        logger.warning("Could not resolve delegation.child_toolsets; denying requested child tools", exc_info=True)
        return False

def _resolve_child_toolsets(
    parent_agent, toolsets: Optional[List[str]], effective_role: str
) -> tuple[List[str], List[str]]:
    """``(enabled_toolsets, disabled_toolsets)`` for a child.

    An absent ``delegation.child_toolsets`` preserves legacy inheritance. When
    configured, it is an exact operator-owned worker surface: parent enables
    and disables are not inherited, and a trusted caller may only request a
    semantic subset. Blocked tools remain blocked in both modes.
    """
    configured = _get_configured_child_toolsets()
    using_worker_policy = configured is not None
    if using_worker_policy:
        if toolsets and _configured_child_toolsets_allow(toolsets):
            child_toolsets = list(toolsets)
        elif toolsets:
            child_toolsets = []
        else:
            child_toolsets = list(configured)
    else:
        # enabled_toolsets=None means "all tools", so derive from loaded tool names.
        parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
        if parent_enabled is not None:
            parent_toolsets = set(parent_enabled)
        elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
            import model_tools
            parent_toolsets = {
                ts for name in parent_agent.valid_tool_names if (ts := model_tools.get_toolset_for_tool(name)) is not None
            }
        else:
            parent_toolsets = set(DEFAULT_TOOLSETS)

        if toolsets:
            expanded_parent = _expand_parent_toolsets(parent_toolsets)
            child_toolsets = [t for t in toolsets if t in expanded_parent]
            if _get_inherit_mcp_toolsets():
                # Append any parent MCP toolsets missing from the narrowed child.
                child_toolsets += [
                    name for name in sorted(parent_toolsets) if _is_mcp_toolset_name(name) and name not in child_toolsets
                ]
        elif parent_agent and parent_enabled is not None:
            child_toolsets = parent_enabled
        else:
            child_toolsets = sorted(parent_toolsets) or DEFAULT_TOOLSETS
    # A route/turn policy is an immutable outer authority boundary. Apply it
    # after the worker-policy choice so explicit worker configuration cannot
    # silently bypass a target-issued hosted-room restriction.
    if (room_ceiling := _bound_room_policy_toolsets()) is not None:
        child_toolsets = [toolset for toolset in child_toolsets if toolset in room_ceiling]
    # ``_strip_blocked_tools`` intentionally removes delegation for ordinary
    # children. Preserve the authorization decision before that role-neutral
    # stripping so an orchestrator gets it back only when its selected surface
    # explicitly contains it (or under legacy inheritance).
    delegation_authorized = not using_worker_policy or "delegation" in child_toolsets
    child_toolsets = _strip_blocked_tools(child_toolsets)

    raw_parent_disabled = None if using_worker_policy else getattr(parent_agent, "disabled_toolsets", None)
    inherited_disabled = (
        [str(name) for name in raw_parent_disabled] if isinstance(raw_parent_disabled, (list, tuple, set)) else []
    )
    if effective_role == "orchestrator" and delegation_authorized:
        inherited_disabled = [name for name in inherited_disabled if name != "delegation"]
        if "delegation" not in child_toolsets:
            child_toolsets.append("delegation")
    child_disabled_toolsets = list(
        dict.fromkeys(inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"])
    )
    return child_toolsets, child_disabled_toolsets
