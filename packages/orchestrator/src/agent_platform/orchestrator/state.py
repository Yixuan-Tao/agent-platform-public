"""共享 AgentState。"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - langgraph 未安装时降级
    def add_messages(left: list, right: list) -> list:
        return (left or []) + list(right)


class AgentState(TypedDict, total=False):
    original_goal: str
    replan_count: int
    messages: Annotated[list[Any], add_messages]
    active_agent: str
    task_plan: str
    context: dict[str, Any]
    memory_refs: dict[str, Any]
    attempts: int
    errors: list[str]
    final_answer: str
    done: bool
    # Plan-and-Execute 模式字段
    steps: list[str]
    current_step: int
    step_results: list[str]
    step_outputs: list[dict[str, Any]]
    mode: str


def initial_state(user_message: str) -> AgentState:
    return {
        "messages": [{"role": "user", "content": user_message}],
        "active_agent": "",
        "task_plan": "",
        "context": {},
        "memory_refs": {},
        "attempts": 0,
        "errors": [],
        "final_answer": "",
        "done": False,
    }
