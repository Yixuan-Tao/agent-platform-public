# Test values are synthetic fixtures for software behavior.
from types import SimpleNamespace
from agent_platform.evaluation.runner import ExperimentRunner
from agent_platform.evaluation.dataset import dataset_hash

async def test_batch_recovery_reuses_existing_run_and_retains_failure(monkeypatch):
    task = {"id":"emc-fixture","domain":"emc","kind":"model_task","split":"development","scenario":"数值差异","goal":"fixture","expected":{"value_a":30,"value_b":32,"unit":"dBµV/m"}}
    run = {"id":"existing","owner":"alice","status":"failed","created_at":0,"budget":{"active_seconds":1},"tool_calls":[{"result":{"evidence_refs":[{"doc_id":"A"}]}}],"result":{"answer":"30 32 dBµV/m","verification":{"status":"pass"}}}
    saved = []
    class Store:
        async def get(self,run_id):
            assert run_id == "existing"
            return run
        async def batch(self,payload,batch_id=None):
            saved.append(payload)
            return "batch"
        async def events(self,*args):
            return []
    class Service:
        store = Store()
        def profiles(self):
            return [{"name":"B0"}]
        def code_version(self):
            return "fixed"
        def model_configuration(self):
            return {"model":"fixed"}
        async def create(self,*args,**kwargs):
            raise AssertionError("Recovery must reuse the saved run")
    async def controls(*args):
        return {"passed":True}
    monkeypatch.setattr("agent_platform.evaluation.runner.dataset",lambda:[task])
    monkeypatch.setattr("agent_platform.evaluation.controls.run_controls",controls)
    payload = {"status":"interrupted","owner":"alice","identity":{"username":"alice"},"code_version":"fixed","dataset_hash":dataset_hash(),"runtime_snapshot":{"model":"fixed"},"profile_snapshots":[{"name":"B0"}],"config":{"split":"development","profiles":["B0"],"repetitions":1,"data_version":"fixed"},"runs":[{"run_id":"existing","task_id":"emc-fixture","profile":"B0","repeat":0}],"results":[]}
    await ExperimentRunner(Service()).execute("batch",payload)
    assert payload["status"] == "completed"
    assert len(payload["results"]) == 1 and not payload["results"][0]["success"]
    assert payload["report"]["B0"]["success_rate"] == 0

async def test_corpus_change_cancels_active_run_retains_failure_and_stops_batch(monkeypatch):
    task = {"id":"emc-fixture","domain":"emc","kind":"model_task","split":"development","scenario":"fixture","goal":"fixture","expected":{}}
    run = {"id":"run","owner":"alice","status":"queued","created_at":0,"budget":{"tokens":32000,"model_requests":8},"tool_calls":[],"result":{}}
    created, checks, cancelled = [], [], []
    class Store:
        async def get(self,*args):
            return run
        async def batch(self,*args):
            return "batch"
        async def cancel(self,*args):
            cancelled.append(args)
            run["status"] = "cancelled"
        async def events(self,run_id,owner,cursor):
            return [{"seq":1,"type":"rag_budget","phase":"reserved","grant":{"request_id":"unknown","tokens":32000}}] if cursor == 0 else []
    class Service:
        store = Store()
        def profiles(self):
            return [{"name":"B0"}]
        def code_version(self):
            return "fixed"
        def model_configuration(self):
            return {"model":"fixed"}
        async def validate_rag_snapshot(self,*args):
            checks.append(1)
            if len(checks) == 3:
                raise ValueError("RAG 语料已变化，请重建基线")
        async def create(self,*args,**kwargs):
            created.append(1)
            return run
    async def controls(*args):
        return {"passed":True}
    monkeypatch.setattr("agent_platform.evaluation.runner.dataset",lambda:[task])
    monkeypatch.setattr("agent_platform.evaluation.controls.run_controls",controls)
    payload = {"owner":"alice","identity":{"username":"alice"},"code_version":"fixed","dataset_hash":dataset_hash(),
               "runtime_snapshot":{"model":"fixed"},"rag_snapshot":{"corpus_hash":"fixed"},"profile_snapshots":[{"name":"B0"}],
               "config":{"split":"development","profiles":["B0"],"repetitions":3,"data_version":"fixed"},"runs":[],"results":[]}
    await ExperimentRunner(Service()).execute("batch",payload)
    assert payload["status"] == "failed" and len(created) == 1 and len(cancelled) == 1
    assert not payload["results_valid_for_comparison"]
    assert len(payload["results"]) == 1 and not payload["results"][0]["success"]
    assert payload["results"][0]["tokens_actual"] == 0 and payload["results"][0]["tokens_estimated"] == 32000
    assert payload["report"]["B0"]["n"] == 1
    assert not payload["promotion"]["B0"]["eligible"]
