"""Capability-boundary tests for configured delegated-child toolsets."""

from types import SimpleNamespace

import pytest
import hermes_cli.config as hc

from tools.delegate_tool_toolsets import _resolve_child_toolsets


def _parent(*, enabled, disabled=()):
    return SimpleNamespace(
        enabled_toolsets=list(enabled),
        disabled_toolsets=list(disabled),
    )


def test_child_inherits_parent_toolsets_when_no_worker_policy(monkeypatch):
    """Existing installs retain inheritance unless an operator configures a policy."""
    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: None
    )

    child, disabled = _resolve_child_toolsets(
        _parent(enabled=["delegation", "file", "terminal"]), None, "leaf"
    )

    assert child == ["file", "terminal"]
    assert "delegation" in disabled


def test_worker_policy_can_grant_explicit_tools_absent_from_parent(monkeypatch):
    """A planning-only parent cannot execute worker tools, but its child can."""
    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets",
        lambda: ["file", "terminal"],
    )
    parent = _parent(
        enabled=["delegation", "todo", "messaging"],
        disabled=["file", "terminal"],
    )

    child, disabled = _resolve_child_toolsets(parent, None, "leaf")

    assert child == ["file", "terminal"]
    assert "file" not in parent.enabled_toolsets
    assert "terminal" not in parent.enabled_toolsets
    assert "file" not in disabled
    assert "terminal" not in disabled
    assert "delegation" in disabled


def test_worker_policy_rejects_unconfigured_requested_toolsets(monkeypatch):
    """A caller cannot escape an operator policy by naming another toolset."""
    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets",
        lambda: ["file"],
    )

    child, _disabled = _resolve_child_toolsets(
        _parent(enabled=["delegation"]), ["terminal"], "leaf"
    )

    assert child == []


def test_worker_policy_cannot_exceed_bound_room_execution_policy(monkeypatch):
    """A target-issued room policy remains the authority ceiling for child tools."""
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy,
        bind_room_execution_policy,
        reset_room_execution_policy,
    )

    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets",
        lambda: ["file", "terminal"],
    )
    policy = RoomExecutionPolicy(
        version=1,
        target_profile="reviewer",
        enabled_toolsets=("bot_room", "delegation", "file"),
        approval_mode="manual",
        max_iterations=1,
        policy_digest="test-policy",
    )
    token = bind_room_execution_policy(policy)
    try:
        child, _disabled = _resolve_child_toolsets(
            _parent(enabled=["delegation"]), None, "leaf"
        )
    finally:
        reset_room_execution_policy(token)

    assert child == ["file"]


def test_lifecycle_request_accepts_worker_policy_not_parent_surface(monkeypatch):
    """The public worker API observes the same configured boundary."""
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService

    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets",
        lambda: ["terminal"],
    )
    parent = _parent(enabled=["delegation"])

    SubagentLifecycleService._validate_request(
        SubagentLaunchRequest(goal="run a bounded command", allowed_toolsets=("terminal",)),
        parent,
    )


def test_lifecycle_request_rejects_toolset_outside_worker_policy(monkeypatch):
    """The public worker API fails closed instead of silently broadening a child."""
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleError,
        SubagentLifecycleService,
    )

    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: ["file"]
    )
    parent = _parent(enabled=["delegation"])

    with pytest.raises(SubagentLifecycleError, match="delegation.child_toolsets"):
        SubagentLifecycleService._validate_request(
            SubagentLaunchRequest(goal="run a bounded command", allowed_toolsets=("terminal",)),
            parent,
        )


def test_lifecycle_rejects_empty_toolset_request_under_worker_policy(monkeypatch):
    """An explicit empty request cannot be reinterpreted as the whole worker policy."""
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleError,
        SubagentLifecycleService,
    )

    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: ["terminal"]
    )

    with pytest.raises(SubagentLifecycleError, match="empty"):
        SubagentLifecycleService._validate_request(
            SubagentLaunchRequest(goal="run nothing", allowed_toolsets=()),
            _parent(enabled=["delegation"]),
        )


@pytest.mark.parametrize("policy", [["file", "*"], ["missing-toolset"], "terminal", None])
def test_malformed_worker_policy_denies_child_tools(monkeypatch, policy):
    """A typo or wildcard cannot silently restore the parent's broad surface."""
    from tools.delegate_tool_config import _get_configured_child_toolsets

    monkeypatch.setattr(
        "tools.delegate_tool_config._load_child_toolset_policy_config", lambda: {"child_toolsets": policy}
    )

    assert _get_configured_child_toolsets() == []


def test_dynamic_worker_toolset_uses_registry_validation(monkeypatch):
    """Live MCP/plugin aliases are valid policy names, not static-name failures."""
    from tools.delegate_tool_config import _get_configured_child_toolsets

    monkeypatch.setattr(
        "tools.delegate_tool_config._load_child_toolset_policy_config", lambda: {"child_toolsets": ["mcp-worker"]}
    )
    monkeypatch.setattr("toolsets.validate_toolset", lambda name: name == "mcp-worker")

    assert _get_configured_child_toolsets() == ["mcp-worker"]


def test_profile_read_failure_denies_instead_of_using_cli_snapshot(monkeypatch):
    """An unreadable active profile cannot fall back to another session's tools."""
    from tools.delegate_tool_config import _get_configured_child_toolsets

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

    def fail_load():
        raise RuntimeError("profile unavailable")

    monkeypatch.setattr("hermes_cli.config.require_readable_config_before_write", fail_load)

    assert _get_configured_child_toolsets() == []


def test_invalid_active_profile_does_not_recover_to_parent_tools(tmp_path, monkeypatch):
    """The config loader's normal defaults recovery is unsafe for authorization."""
    from tools.delegate_tool_config import _get_configured_child_toolsets

    home = tmp_path / "invalid-profile"
    home.mkdir()
    (home / "config.yaml").write_text("delegation: [not-a-mapping\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    hc._LOAD_CONFIG_CACHE.clear()
    hc._RAW_CONFIG_CACHE.clear()

    assert _get_configured_child_toolsets() == []


def test_unrelated_schema_error_cannot_hide_explicit_deny_policy(tmp_path, monkeypatch):
    """Raw policy reads never accept the merged loader's defaults recovery."""
    from tools.delegate_tool_config import _get_configured_child_toolsets

    home = tmp_path / "schema-error-profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "max_turns: 10\nagent: invalid\ndelegation:\n  child_toolsets: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    hc._LOAD_CONFIG_CACHE.clear()
    hc._RAW_CONFIG_CACHE.clear()

    assert _get_configured_child_toolsets() == []


def test_worker_policy_does_not_grant_orchestrator_delegation(monkeypatch):
    """Depth-derived orchestration cannot escape an empty worker allowlist."""
    monkeypatch.setattr("tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: [])

    child, disabled = _resolve_child_toolsets(_parent(enabled=["delegation"]), None, "orchestrator")

    assert child == []
    assert "delegation" not in child
    assert "delegation" not in disabled


def test_worker_policy_restores_explicit_orchestrator_delegation(monkeypatch):
    """The role may retain delegation only when the selected policy explicitly allows it."""
    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: ["delegation", "file"]
    )

    orchestrator, _disabled = _resolve_child_toolsets(
        _parent(enabled=["delegation"]), None, "orchestrator"
    )
    leaf, leaf_disabled = _resolve_child_toolsets(
        _parent(enabled=["delegation"]), None, "leaf"
    )

    assert orchestrator == ["file", "delegation"]
    assert leaf == ["file"]
    assert "delegation" in leaf_disabled


def test_lifecycle_narrowing_does_not_regrant_orchestrator_delegation(monkeypatch):
    """An API caller that requests file cannot inherit policy delegation transitively."""
    monkeypatch.setattr(
        "tools.delegate_tool_toolsets._get_configured_child_toolsets", lambda: ["delegation", "file"]
    )

    child, _disabled = _resolve_child_toolsets(
        _parent(enabled=["delegation"]), ["file"], "orchestrator"
    )

    assert child == ["file"]


def test_worker_policy_uses_only_the_active_profile(tmp_path, monkeypatch):
    """Real config loads never leak an operational allowlist across profiles."""
    parent = _parent(enabled=["delegation"], disabled=["file", "terminal"])
    first = tmp_path / "first-profile"
    second = tmp_path / "second-profile"
    for profile, policy in ((first, "[terminal]"), (second, "[file]")):
        profile.mkdir()
        (profile / "config.yaml").write_text(
            "model:\n  default: test-model\n"
            f"delegation:\n  child_toolsets: {policy}\n",
            encoding="utf-8",
        )

    monkeypatch.setenv("HERMES_HOME", str(first))
    hc._LOAD_CONFIG_CACHE.clear()
    assert _resolve_child_toolsets(parent, None, "leaf")[0] == ["terminal"]

    monkeypatch.setenv("HERMES_HOME", str(second))
    hc._LOAD_CONFIG_CACHE.clear()
    assert _resolve_child_toolsets(parent, None, "leaf")[0] == ["file"]
