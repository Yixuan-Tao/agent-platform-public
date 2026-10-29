"""Optional OTLP/HTTP export. Durable local events remain authoritative on failure."""
import asyncio
import json
import time
import uuid
import httpx

class LangfuseExporter:
    def __init__(self, settings):
        self.settings = settings
        self.queue = asyncio.Queue(maxsize=256)
        self.errors = 0
        self.starts = {}
        self.task = None

    def submit(self, run_id, kind, payload):
        # Export metadata only: no prompts, raw reasoning, tool arguments or document content.
        if kind not in {"model", "tool", "agent", "router", "verification", "status"}:
            return
        safe = {k: v for k, v in payload.items() if k in {"phase", "model", "duration", "usage", "usage_source", "cost", "tool", "agent", "status", "error", "tool_call_id", "model_call_id", "service", "rag_request_id"}}
        try:
            self.queue.put_nowait((run_id, kind, safe, time.time_ns()))
        except asyncio.QueueFull:
            self.errors += 1

    def span(self, run_id, kind, payload, now):
        key = (run_id, payload.get("model_call_id") or payload.get("tool_call_id") or kind)
        if payload.get("phase") == "start":
            self.starts[key] = now
            return None
        started = self.starts.pop(key, now)
        if payload.get("service") == "rag" and payload.get("duration") is not None:
            # Delegated receipts arrive together; use their measured duration, not queue spacing.
            started = now - int(max(0, payload["duration"]) * 1e9)
        attributes = {"run_id": run_id, "langfuse.trace.name": "agent-platform-run",
                      "langfuse.observation.metadata": json.dumps(payload, ensure_ascii=False),
                      "langfuse.observation.type": "generation" if kind == "model" else "tool" if kind == "tool" else "span"}
        if payload.get("model"):
            attributes["gen_ai.request.model"] = payload["model"]
        usage = payload.get("usage", {})
        if payload.get("usage_source") == "actual":
            for source, target in (("prompt_tokens", "gen_ai.usage.input_tokens"), ("completion_tokens", "gen_ai.usage.output_tokens")):
                if usage.get(source) is not None:
                    attributes[target] = int(usage[source])
        return {"traceId": run_id[:32], "spanId": uuid.uuid4().hex[:16],
                "name": kind + ":" + str(payload.get("tool") or payload.get("model") or payload.get("status") or "event"),
                "startTimeUnixNano": str(started), "endTimeUnixNano": str(now),
                "attributes": [{"key": k, "value": {"intValue": str(v)} if isinstance(v, int) else {"stringValue": str(v)}} for k, v in attributes.items()]}

    async def serve(self):
        async with httpx.AsyncClient(timeout=3) as client:
            while True:
                run_id, kind, payload, now = await self.queue.get()
                span = self.span(run_id, kind, payload, now)
                try:
                    if span:
                        response = await client.post(self.settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces",
                            auth=(self.settings.langfuse_public_key, self.settings.langfuse_secret_key),
                            headers={"x-langfuse-ingestion-version": "4"},
                            json={"resourceSpans": [{"scopeSpans": [{"scope": {"name": "agent-platform"}, "spans": [span]}]}]})
                        response.raise_for_status()
                except Exception:
                    self.errors += 1
                finally:
                    self.queue.task_done()

    def start(self):
        self.task = asyncio.create_task(self.serve())

    async def close(self):
        try:
            await asyncio.wait_for(self.queue.join(), 5)
        except asyncio.TimeoutError:
            pass
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
