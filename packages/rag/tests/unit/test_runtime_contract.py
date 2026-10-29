# Test values are synthetic fixtures for software behavior.
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.runtime_contract import RuntimeContractMiddleware
from utils.runtime_budget import RequestBudget, current_budget, model_post, model_chat, submit_context, RemoteBudgetExceeded
from utils.corpus_snapshot import runtime_snapshot, collection_hash, evidence_hash


def grant(**changes):
    return {"version": 1, "request_id": uuid.uuid4().hex, "run_id": uuid.uuid4().hex,
            "model_requests": 8, "tokens": 32000, "deadline": time.time()+60, "corpus_hash": "a"*64, **changes}


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path))
    token = current_budget.set(None)
    yield
    current_budget.reset(token)


def test_retry_each_attempt_and_missing_usage_stays_estimated(monkeypatch):
    from utils.retry import httpx_retry_post
    replies = [httpx.Response(500, json={}), httpx.Response(200, json={"usage": {"total_tokens": 7}})]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr("utils.retry.time.sleep", lambda *a: None)
    budget = RequestBudget(grant())
    current_budget.set(budget)
    httpx_retry_post(max_retries=1)("http://fixture/embeddings", json={"model": "embedding", "input": "中文内容"})
    report = budget.report("a"*64)
    assert report["model_requests"] == 2
    assert [c["usage_source"] for c in report["calls"]] == ["estimated", "actual"]
    assert report["tokens"] == report["calls"][0]["tokens_reserved"] + 7
    assert report["calls"][0]["error"] == "HTTP_500"


def test_parallel_submissions_share_atomic_grant_before_network(monkeypatch):
    sent = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (sent.append(k) or httpx.Response(200, json={"usage": {"total_tokens": 1}})))
    budget = RequestBudget(grant(model_requests=1))
    current_budget.set(budget)
    with ThreadPoolExecutor(2) as pool:
        futures = [submit_context(pool, model_post, "http://fixture/embeddings", json={"model": "m", "input": "q"}) for _ in range(2)]
        failures = 0
        for future in futures:
            try:
                future.result()
            except RemoteBudgetExceeded:
                failures += 1
    assert len(sent) == 1 and failures == 1
    assert budget.error == "RAG_BUDGET_EXHAUSTED"


def test_chat_disables_sdk_retries_limits_output_and_records_overrun():
    settings, calls = [], []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage={"total_tokens": 5000})
    client = SimpleNamespace(with_options=lambda **kw: (settings.append(kw) or SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    budget = RequestBudget(grant(tokens=4800))
    current_budget.set(budget)
    model_chat(client, model="model", messages=[{"role": "user", "content": "问题"}], stream=False)
    assert settings[0]["max_retries"] == 0 and calls[0]["max_tokens"] == 4096
    assert budget.tokens == 5000 and budget.error == "RAG_BUDGET_EXHAUSTED"
    with pytest.raises(RemoteBudgetExceeded):
        model_chat(client, model="model", messages=[])
    assert len(calls) == 1


def make_app(monkeypatch, operation):
    app = FastAPI()
    app.state.runtime_snapshot_provider = lambda: {"version": 1, "corpus_hash": "a"*64}
    app.add_middleware(RuntimeContractMiddleware)
    @app.post("/tools/retrieve")
    def retrieve():
        return operation(app)
    return app


def test_asgi_receipt_passes_threadpool_and_persistent_replay_is_rejected(monkeypatch):
    sent = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (sent.append(1) or httpx.Response(200, json={"usage": {"total_tokens": 12}})))
    def operation(app):
        model_post("http://fixture/embeddings", json={"model": "m", "input": "q"})
        return {"results": [{"text": "证据", "doc_id": "A"}]}
    app = make_app(monkeypatch, operation)
    request = grant()
    for index in range(2):
        # Recreate TestClient; replay protection is SQLite-backed, not tied to a request context.
        with TestClient(app) as client:
            response = client.post("/tools/retrieve", headers={"X-Agent-Runtime": json.dumps(request)}, json={"query": "q"})
        if index == 0:
            assert response.status_code == 200
            assert response.json()["_runtime"]["tokens"] == 12
            assert response.json()["_runtime"]["model_requests"] == 1
        else:
            assert response.status_code == 409
            assert response.json()["error_code"] == "RAG_REQUEST_REPLAY"
    assert len(sent) == 1


def test_corpus_mismatch_stops_before_model_and_changes_during_call_discard_result(monkeypatch):
    sent = []
    def operation(app):
        sent.append(1)
        app.state.runtime_snapshot_provider = lambda: {"version": 1, "corpus_hash": "b"*64}
        return {"results": [{"text": "old evidence"}]}
    app = make_app(monkeypatch, operation)
    with TestClient(app) as client:
        before = client.post("/tools/retrieve", headers={"X-Agent-Runtime": json.dumps(grant(corpus_hash="b"*64))})
        assert before.status_code == 409 and not sent
        after = client.post("/tools/retrieve", headers={"X-Agent-Runtime": json.dumps(grant())})
        assert after.status_code == 409 and sent == [1]
        assert "results" not in after.json()
        assert after.json()["_runtime"]["corpus_hash"] == "b"*64


def test_denial_cannot_be_hidden_by_legacy_fallback(monkeypatch):
    sent = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (sent.append(1) or httpx.Response(200, json={})))
    def operation(app):
        try:
            model_post("http://fixture/embeddings", json={"model": "m", "input": "文本"*1000})
        except Exception:
            pass
        return {"results": [{"text": "fallback success"}]}
    with TestClient(make_app(monkeypatch, operation)) as client:
        response = client.post("/tools/retrieve", headers={"X-Agent-Runtime": json.dumps(grant(tokens=10))})
    assert response.status_code == 429 and not sent
    assert response.json()["_runtime"]["error"] == "RAG_BUDGET_EXHAUSTED"


class Collection:
    metadata = {"hnsw:space": "cosine"}
    def __init__(self, rows):
        self.rows = rows
    def count(self):
        return len(self.rows)
    def get(self, limit, offset, include):
        rows = self.rows[offset:offset+limit]
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                "metadatas": [r[2] for r in rows], "embeddings": [r[3] for r in rows]}


def test_logical_snapshot_detects_equal_count_text_units_versions_and_vector_changes(tmp_path):
    rows = [["chunk", "30 dBµV/m", {"version": "2020", "page": 12}, [1., 0.]]]
    chroma = SimpleNamespace(is_connected=True, _db_path=str(tmp_path), _doc_collection=Collection(rows), _image_collection=Collection([]))
    original = runtime_snapshot(chroma)
    assert original == runtime_snapshot(chroma)
    rows[0][1] = "31 dBµV/m"
    assert original["corpus_hash"] != runtime_snapshot(chroma)["corpus_hash"]
    rows[0][1] = "30 dBµV/m"
    rows[0][2]["version"] = "2021"
    assert original["corpus_hash"] != runtime_snapshot(chroma)["corpus_hash"]
    rows[0][2]["version"] = "2020"
    rows[0][3] = [0., 1.]
    assert original["corpus_hash"] != runtime_snapshot(chroma)["corpus_hash"]


def test_collection_order_is_not_a_data_version():
    rows = [["b", "B", {}, [1.]], ["a", "A", {}, [0.]]]
    assert collection_hash(Collection(rows), 1) == collection_hash(Collection(list(reversed(rows))), 1)


def test_published_evidence_is_frozen_while_draft_changes_are_ignored(tmp_path):
    from utils.evidence_store import EvidenceStore, EvidenceItem
    store = EvidenceStore(tmp_path / "evidence.db")
    draft = store.create_version("fixture", "emc")
    original = evidence_hash(store)
    store.add_items(draft["id"], [EvidenceItem(subject_id="A", domain="emc", field_name="限值", value_raw="30", unit="dBµV/m")])
    assert original == evidence_hash(store)
    store.publish(draft["id"])
    assert original != evidence_hash(store)


def test_run_and_corpus_scoped_cache_never_uses_prior_run_answer():
    from pipeline.query import QueryPipeline
    original = QueryPipeline._cache_key("q", 5, True, "zh", "")
    first = grant()
    current_budget.set(RequestBudget(first))
    key = QueryPipeline._cache_key("q", 5, True, "zh", "")
    assert key != original
    current_budget.set(RequestBudget(grant()))
    assert key != QueryPipeline._cache_key("q", 5, True, "zh", "")
    current_budget.set(RequestBudget({**first, "request_id": uuid.uuid4().hex}))
    assert key == QueryPipeline._cache_key("q", 5, True, "zh", "")

def test_snapshot_detects_image_content_and_settings_without_exposing_credentials(tmp_path,monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    image = root / "fixture.png"
    image.write_bytes(b"fixture-image-v1")
    chroma = SimpleNamespace(is_connected=True,_db_path=str(tmp_path),_doc_collection=Collection([]),_image_collection=Collection([]))
    monkeypatch.setenv("ONEAPI_API_KEY","private-fixture-credential")
    first = runtime_snapshot(chroma)
    assert "private-fixture-credential" not in json.dumps(first)
    image.write_bytes(b"fixture-image-v2")
    assert first["corpus_hash"] != runtime_snapshot(chroma)["corpus_hash"]
    second = runtime_snapshot(chroma)
    monkeypatch.setenv("REQUERY_ENABLED","false")
    off = runtime_snapshot(chroma)
    monkeypatch.setenv("REQUERY_ENABLED","true")
    assert off["corpus_hash"] != runtime_snapshot(chroma)["corpus_hash"]

def test_snapshot_rejects_changes_during_scan(tmp_path):
    marker = tmp_path / "chroma.sqlite3"
    marker.write_bytes(b"initial")
    class ChangingCollection(Collection):
        def get(self,**kwargs):
            marker.write_bytes(marker.read_bytes()+b"changed")
            return super().get(**kwargs)
    chroma = SimpleNamespace(is_connected=True,_db_path=str(tmp_path),_doc_collection=ChangingCollection([["a","A",{},[0.]]]),_image_collection=Collection([]))
    with pytest.raises(RuntimeError,match="changed"):
        runtime_snapshot(chroma)

def test_logical_snapshot_with_real_local_chroma_and_equal_count_update(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ,"PYTHONPATH":str(root),"LLM_API_KEY":"","ONEAPI_API_KEY":"","CHAT_API_KEY":"","DASHSCOPE_API_KEY":"",
                   "RUNTIME_DIR":str(tmp_path),"PYTHONIOENCODING":"utf-8"}
    result = subprocess.run([sys.executable,str(root / "tests/fixtures/chroma_snapshot_probe.py"),str(tmp_path / "chroma")],
                            cwd=root,env=environment,capture_output=True,text=True,encoding="utf-8",timeout=20,
                            **({"creationflags":subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout.strip()) == {"count":1,"stable_before":True,"changed_after_equal_count_update":True}

def test_real_openai_sdk_wire_usage_and_hidden_retry_disabled():
    from openai import OpenAI, APIStatusError
    requests = []
    fail = [False]
    def provider(request):
        requests.append(json.loads(request.content))
        if fail[0]:
            return httpx.Response(500,json={"error":{"message":"fixture","type":"server_error"}})
        return httpx.Response(200,json={"id":"fixture","object":"chat.completion","created":0,"model":"m",
             "choices":[{"index":0,"message":{"role":"assistant","content":"fixture"},"finish_reason":"stop"}],
             "usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}})
    budget = RequestBudget(grant())
    current_budget.set(budget)
    with OpenAI(api_key="fixture",base_url="http://provider.invalid/v1",max_retries=3,
                http_client=httpx.Client(transport=httpx.MockTransport(provider))) as client:
        model_chat(client,model="m",messages=[{"role":"user","content":"q"}])
        assert budget.report("a"*64)["calls"][0]["usage"] == {"total_tokens":4}
        assert requests[0]["max_tokens"] == 4096
        fail[0] = True
        with pytest.raises(APIStatusError):
            model_chat(client,model="m",messages=[{"role":"user","content":"q"}])
        assert len(requests) == 2 and len(budget.calls) == 2

def test_snapshot_cannot_claim_changed_disk_code_is_loaded(tmp_path,monkeypatch):
    chroma = SimpleNamespace(is_connected=True,_db_path=str(tmp_path),_doc_collection=Collection([]),_image_collection=Collection([]))
    monkeypatch.setattr("utils.corpus_snapshot.executable_hash",lambda:"changed-source")
    with pytest.raises(RuntimeError,match="restart"):
        runtime_snapshot(chroma)
