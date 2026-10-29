"""Usage/budget hooks for both OpenAI-client and LangChain model paths."""
import json
import time
import uuid
from langchain_core.callbacks import AsyncCallbackHandler
from agent_platform.core.execution import execution_context, estimate_tokens, reserve, emit


async def begin(messages, model, output_limit=4096):
    await reserve("model_requests")
    reserved = estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str)) + output_limit
    await reserve("tokens", reserved)
    call_id = uuid.uuid4().hex
    await emit("model", phase="start", model=model, tokens_reserved=reserved, model_call_id=call_id)
    return reserved, time.monotonic(), call_id


async def end(ticket, model, usage=None, error=None):
    reserved, started, call_id = ticket
    total = (usage or {}).get("total_tokens")
    ctx = execution_context.get()
    if total is not None and ctx and ctx.settle:
        await ctx.settle("tokens", int(total) - reserved)
    await emit("model", phase="end", model=model, duration=time.monotonic() - started, model_call_id=call_id,
               usage=usage or {"total_tokens": reserved}, usage_source="actual" if total is not None else "estimated",
               cost=None, error=type(error).__name__ if error else None)


class ModelTelemetry(AsyncCallbackHandler):
    raise_error = True
    def __init__(self, model, output_limit=4096):
        self.model, self.output_limit = model, output_limit
        self._tickets = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._tickets[run_id] = await begin(messages, self.model, self.output_limit)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        ticket = self._tickets.pop(run_id, None)
        if ticket:
            usage = (response.llm_output or {}).get("token_usage")
            if not usage and response.generations:
                message = response.generations[0][0].message
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    usage = {"prompt_tokens": metadata.get("input_tokens"), "completion_tokens": metadata.get("output_tokens"), "total_tokens": metadata.get("total_tokens")}
            await end(ticket, self.model, usage)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        ticket = self._tickets.pop(run_id, None)
        if ticket:
            await end(ticket, self.model, error=error)
