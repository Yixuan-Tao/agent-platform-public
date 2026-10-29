"""Read-only resource tools and explicit user-input interrupt for durable runs."""
import base64
from agent_platform.core.execution import execution_context
from agent_platform.tools.base import BaseTool
from agent_platform.tools.result import ToolResult


class ReadArtifactTool(BaseTool):
    name = "read_run_artifact"
    description = "按产物 ID 分页读取本任务原始证据；不能读取其他任务。"
    args_schema = {"type": "object", "properties": {"artifact_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 16000}}, "required": ["artifact_id"], "additionalProperties": False}
    async def execute(self, artifact_id, offset=0, length=8000):
        ctx = execution_context.get()
        item = await ctx.resources["store"].read_artifact(ctx.run_id, ctx.actor, artifact_id)
        text = base64.b64decode(item["content_base64"]).decode("utf-8")
        return ToolResult(status="success", data={"content": text[offset:offset+length], "next_offset": offset+length if offset+length < len(text) else None})


class ReadSkillTool(BaseTool):
    name = "read_skill_resource"
    description = "按需读取本任务已选择技能的说明或模板，不执行脚本。"
    args_schema = {"type": "object", "properties": {"skill": {"type": "string"}, "path": {"type": "string"}}, "required": ["skill", "path"], "additionalProperties": False}
    async def execute(self, skill, path):
        ctx = execution_context.get()
        if skill not in ctx.resources["selected_skills"]:
            return ToolResult(status="failed", error_code="FORBIDDEN")
        return ToolResult(status="success", data=ctx.resources["skills"].skills[skill].resource(path))


class RequestInputTool(BaseTool):
    name = "request_run_input"
    description = "缺少文档版本、故障信息等必要条件时暂停任务，请用户补充；返回真实用户输入。"
    args_schema = {"type": "object", "properties": {"question": {"type": "string", "minLength": 1}}, "required": ["question"], "additionalProperties": False}
    async def execute(self, question):
        from langgraph.types import interrupt
        value = interrupt({"kind": "input", "question": question})
        return ToolResult(status="success", data=value)


def runtime_tools():
    return [ReadArtifactTool(), ReadSkillTool(), RequestInputTool()]
