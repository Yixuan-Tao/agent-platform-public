# Test values are synthetic fixtures for software behavior.
"""Loopback test RAG server: real HTTP contract and fake provider responses only."""
from types import SimpleNamespace
import sys

from fastapi import FastAPI
import httpx
import uvicorn

from api.runtime_contract import RuntimeContractMiddleware, router as runtime_router
from utils.runtime_budget import model_post, model_chat, submit_context
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
app.add_middleware(RuntimeContractMiddleware)
app.include_router(runtime_router)
revision = ["a"*64]
app.state.runtime_snapshot_provider = lambda: {"version":1,"corpus_hash":revision[0]}


def fake_provider_post(url, **kwargs):
    assert url == "http://provider.invalid/embeddings"
    return httpx.Response(200,json={"usage":{"total_tokens":11}})


httpx.post = fake_provider_post
fake_client = SimpleNamespace(with_options=lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
    create=lambda **kwargs: SimpleNamespace(usage={"total_tokens":17})
))))


@app.get("/health")
def health():
    return {"status":"fixture"}


@app.post("/tools/retrieve")
def retrieve(body: dict):
    # Real ThreadPoolExecutor propagation and the same two provider entry points as RAG.
    with ThreadPoolExecutor(2) as pool:
        calls = [submit_context(pool,model_post,"http://provider.invalid/embeddings",json={"model":"embed","input":"q"}),
                 submit_context(pool,model_chat,fake_client,model="generator",messages=[{"role":"user","content":"q"}],stream=False)]
        for call in calls:
            call.result()
    if body["query"] == "change-index":
        revision[0] = "b"*64
    return {"results":[{"doc_id":"A","chunk_id":"A-1","source":"fixture.pdf","text":"30 dBµV/m"}]}


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=int(sys.argv[1]),log_level="error")
