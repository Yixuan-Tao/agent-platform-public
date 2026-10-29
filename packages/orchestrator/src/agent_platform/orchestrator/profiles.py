"""Immutable experiment profiles. Secrets and request identity are never profile fields."""
from __future__ import annotations
from copy import deepcopy
from agent_platform.core.execution import fingerprint

DEFAULT_BUDGET = {"agent_calls": 8, "model_requests": 32, "tool_attempts": 32,
                  "tokens": 100000, "active_seconds": 600, "plan_steps": 12,
                  "replans": 2, "tool_concurrency": 3, "agent_concurrency": 2}
_CONFIGS = {
    "B0": ("react", "none", "none"), "B1": ("react", "static", "none"),
    "B2": ("react", "progressive", "none"), "C1": ("react", "none", "bounded"),
    "C2": ("react", "progressive", "bounded"), "P1": ("plan_execute", "progressive", "bounded"),
    "M1": ("collaboration", "progressive", "bounded"),
}

def profiles(model: str, skill_versions: list[dict] = (), *, capabilities=()) -> list[dict]:
    results = []
    for name, (strategy, skill, context) in _CONFIGS.items():
        value = dict(name=name, version=1, model=model, temperature=0, execution_strategy=strategy,
                     skill_policy=skill, context_policy=context, memory_policy="private_verified_hash",
                     tool_policy="fixed_authorized", budget=deepcopy(DEFAULT_BUDGET),
                     skills=deepcopy(list(skill_versions)), model_window=32768)
        value["version_hash"] = fingerprint(value)
        results.append(value)
    for name, capability, policy, field in (("X1", "python_container", "python_container", "tool_policy"),
                                           ("S1", "semantic_memory", "private_verified_semantic", "memory_policy"),
                                           ("D1", "tool_discovery", "discover_authorized", "tool_policy")):
        if capability in capabilities:
            value = deepcopy(next(p for p in results if p["name"] == "C2"))
            value.update(name=name, experimental=True)
            value[field] = policy
            value.pop("version_hash")
            value["version_hash"] = fingerprint(value)
            results.append(value)
    return results
