"""Lossless tool result with a compatibility text projection."""
from __future__ import annotations
import json
from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    status: Literal["success", "failed", "partial", "unknown"]
    data: Any = None
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    external_refs: list[dict[str, Any]] = Field(default_factory=list)
    error_code: str = ""
    retryable: bool = False

    def text(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, default=str)


class StructuredToolText(str):
    """Legacy readable projection retaining the complete machine-readable contract."""
    def __new__(cls, text: str, *, status="success", data=None, evidence_refs=(), external_refs=(), error_code="", retryable=False):
        value = super().__new__(cls, text)
        value.tool_result = ToolResult(status=status, data=data, evidence_refs=list(evidence_refs),
            external_refs=list(external_refs), error_code=error_code, retryable=retryable)
        return value
