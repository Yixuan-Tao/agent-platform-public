# Test values are synthetic fixtures for software behavior.
import json
from types import SimpleNamespace
import httpx
import pytest

from agent_platform.core.execution import execution_context, ExecutionContext, BudgetExceeded, reserve
from agent_platform.tools.rag_contract import post, validated_report, RagContractError


@pytest.fixture
def context():
    grants, receipts = [], []
    class Store:
        async def rag_grant(self, run_id, fence, request_id, *args):
            value = {"version":1, "request_id":request_id, "model_requests":8, "tokens":32000, "corpus_hash":"a"*64}
            grants.append(value)
            return value
        async def rag_settle(self, run_id, fence, request_id, report, *args):
            receipts.append(report)
    resources = {"store":Store(), "fence":("worker",1), "rag_snapshot":{"corpus_hash":"a"*64}}
    ctx = ExecutionContext(actor="alice",run_id="run",resources=resources)
    token = execution_context.set(ctx)
    yield ctx, grants, receipts
    execution_context.reset(token)


SETTINGS = SimpleNamespace(rag_service_url="http://rag", rag_service_token="fixture")


def receipt(request_id, tokens=7, **changes):
    call = {"model_call_id":"model-call", "model":"embedding", "tokens_reserved":10,
            "usage":{"total_tokens":tokens}, "usage_source":"actual", "duration":.1, "error":None}
    return {"version":1, "request_id":request_id,"model_requests":1,"tokens":tokens,"calls":[call],"corpus_hash":"a"*64,"error":None,**changes}


async def test_rpc_receipt_is_bound_and_trace_removed_from_model_payload(context):
    ctx, grants, receipts = context
    async def send(request):
        grant = json.loads(request.headers["X-Agent-Runtime"])
        assert request.headers["X-Local-Token"] == "fixture"
        return httpx.Response(200,json={"results":[{"text":"证据"}],"_runtime":receipt(grant["request_id"])})
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        data = await post(client,SETTINGS,"/tools/retrieve",{"query":"q"})
    assert data == {"results":[{"text":"证据"}]}
    assert len(grants) == 1 and receipts[0]["tokens"] == 7


@pytest.mark.parametrize("mode",["timeout","missing","forged","duplicate"])
async def test_unknown_or_invalid_receipt_keeps_entire_reservation(context,mode):
    ctx, grants, receipts = context
    def send(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("lost",request=request)
        grant = json.loads(request.headers["X-Agent-Runtime"])
        if mode == "missing":
            return httpx.Response(200,json={"results":[]})
        value = receipt("another-request" if mode == "forged" else grant["request_id"])
        if mode == "duplicate":
            value["calls"] *= 2
            value["model_requests"], value["tokens"] = 2,14
        return httpx.Response(200,json={"_runtime":value})
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with pytest.raises((httpx.ReadTimeout,RagContractError)):
            await post(client,SETTINGS,"/tools/retrieve",{})
    assert len(grants) == 1 and receipts == [None]


async def test_corpus_change_settles_real_consumption_and_invalidates_next_call(context):
    ctx, grants, receipts = context
    def send(request):
        grant = json.loads(request.headers["X-Agent-Runtime"])
        return httpx.Response(409,json={"error_code":"RAG_CORPUS_CHANGED","_runtime":receipt(grant["request_id"],corpus_hash="b"*64)})
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with pytest.raises(RagContractError):
            await post(client,SETTINGS,"/tools/retrieve",{})
    assert receipts[0]["tokens"] == 7
    with pytest.raises(RuntimeError,match="语料"):
        await reserve("model_requests")


async def test_remote_budget_denial_does_not_turn_into_tool_success(context):
    ctx, grants, receipts = context
    def send(request):
        grant = json.loads(request.headers["X-Agent-Runtime"])
        return httpx.Response(429,json={"_runtime":receipt(grant["request_id"],error="RAG_BUDGET_EXHAUSTED")})
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with pytest.raises(BudgetExceeded):
            await post(client,SETTINGS,"/tools/retrieve",{})
    assert receipts[0]["tokens"] == 7


def test_receipt_sanitizes_untrusted_extra_fields():
    grant = {"request_id":"id", "model_requests":8, "tokens":32000}
    value = receipt("id")
    value["calls"][0]["prompt"] = "must not export"
    value["calls"][0]["usage"]["private"] = "must not export"
    clean = validated_report({"_runtime":value},grant)
    assert "private" not in clean["calls"][0]["usage"] and "prompt" not in clean["calls"][0]
