"""Manually triggered batches; immutable tasks/profiles and simulated side effects."""
import asyncio
import time
from agent_platform.evaluation.dataset import dataset, dataset_hash
from agent_platform.evaluation.metrics import summarize, promotion, usage_totals
from agent_platform.orchestrator.runtime_store import ExperimentRecord, TERMINAL


class ExperimentRunner:
    def __init__(self, service):
        self.service, self.tasks = service, {}

    async def create(self, identity, config):
        available = {p["name"]: p for p in self.service.profiles()}
        if len(set(config["profiles"])) != len(config["profiles"]):
            raise ValueError("一个批次中的配置不能重复")
        if any(name not in available for name in config["profiles"]) or not config["profiles"]:
            raise ValueError("未知实验配置")
        if config["split"] not in {"development", "validation", "holdout"}:
            raise ValueError("未知数据划分")
        if not config["data_version"] or config["data_version"] == "unversioned":
            raise ValueError("实验必须固定数据版本")
        payload = {"owner": identity["username"], "status": "running", "config": config,
                   "dataset_hash": dataset_hash(), "code_version": self.service.code_version(),
                   "profile_snapshots": [available[n] for n in config["profiles"]], "runs": [], "results": [],
                   "runtime_snapshot": self.service.model_configuration(),
                   "controls": "not_run", "created_at": time.time(), "identity": identity}
        payload["rag_snapshot"] = await self.service.rag_snapshot()
        batch_id = await self.service.store.batch(payload)
        self.tasks[batch_id] = asyncio.create_task(self.execute(batch_id, payload))
        return {"id": batch_id, **payload}

    async def resume(self, batch_id, identity):
        payload = await self.service.store.entry(ExperimentRecord, batch_id)
        if not payload or payload["owner"] != identity["username"]:
            raise KeyError("experiment not found")
        if payload["status"] not in {"interrupted", "failed"}:
            raise ValueError("实验当前不可恢复")
        self.validate_snapshot(payload)
        await self.service.validate_rag_snapshot(payload.get("rag_snapshot"))
        payload = {**payload, "status": "running", "identity": identity}
        payload.pop("error", None)
        await self.service.store.batch(payload, batch_id)
        self.tasks[batch_id] = asyncio.create_task(self.execute(batch_id, payload))
        return {"id": batch_id, **payload}

    def validate_snapshot(self, payload):
        current = {p["name"]: p for p in self.service.profiles()}
        if self.service.code_version() != payload["code_version"] or dataset_hash() != payload["dataset_hash"] or payload.get("runtime_snapshot") != self.service.model_configuration() or any(current.get(p["name"]) != p for p in payload["profile_snapshots"]):
            raise ValueError("代码、语料或运行配置已变化，请结束批次并重建基线")

    async def result_row(self, run, task, profile, current, started, success=False, failure_category=None):
        events, cursor = [], 0
        while page := await self.service.store.events(run["id"], run["owner"], cursor):
            events.extend(page)
            cursor = page[-1]["seq"]
        return {"run_id":run["id"],"task_id":task["id"],"profile":profile,"success":success,
                "latency":current["budget"].get("active_seconds",time.monotonic()-started),
                "wall_time":max(0,time.time()-current["created_at"]), **usage_totals(events),
                "model_requests":current["budget"].get("model_requests",0), "tool_attempts":current["budget"].get("tool_attempts",0),
                "failure_category":failure_category or current["result"].get("verification",{}).get("reason") or current["status"]}

    async def execute(self, batch_id, payload):
        try:
            from agent_platform.evaluation.controls import run_controls
            self.validate_snapshot(payload)
            if payload.get("rag_snapshot") is not None:
                await self.service.validate_rag_snapshot(payload["rag_snapshot"])
            payload["controls"] = await run_controls(batch_id)
            await self.service.store.batch(payload, batch_id)
            if not payload["controls"]["passed"]:
                raise ValueError("固定治理控制未全部通过，已停止真实模型实验")
            selected = [t for t in dataset() if t["split"] == payload["config"]["split"] and t["kind"] == "model_task"]
            for profile in payload["config"]["profiles"]:
                for task in selected:
                    for repeat in range(payload["config"]["repetitions"]):
                        self.validate_snapshot(payload)
                        if payload.get("rag_snapshot") is not None:
                            await self.service.validate_rag_snapshot(payload["rag_snapshot"])
                        previous = next((r for r in payload["runs"] if r["task_id"] == task["id"] and r["profile"] == profile and r["repeat"] == repeat), None)
                        if previous and any(r["run_id"] == previous["run_id"] for r in payload["results"]):
                            continue
                        if previous:
                            run = await self.service.store.get(previous["run_id"])
                        else:
                            run = await self.service.create(payload["identity"], task["goal"], profile=profile,
                                                            data_version=payload["config"]["data_version"], completion={"fixture": task, "batch_id": batch_id},
                                                            rag_snapshot=payload.get("rag_snapshot"))
                            payload["runs"].append({"run_id": run["id"], "task_id": task["id"], "profile": profile, "repeat": repeat})
                        await self.service.store.batch(payload, batch_id)
                        started = time.monotonic()
                        last_corpus_check = 0
                        while True:
                            if payload.get("rag_snapshot") is not None and time.monotonic()-last_corpus_check >= 5:
                                try:
                                    await self.service.validate_rag_snapshot(payload["rag_snapshot"])
                                    last_corpus_check = time.monotonic()
                                except Exception:
                                    await self.service.store.cancel(run["id"], run["owner"])
                                    current = await self.service.store.get(run["id"])
                                    payload["results"].append(await self.result_row(run,task,profile,current,started,failure_category="corpus_changed_or_unavailable"))
                                    raise
                            current = await self.service.store.get(run["id"])
                            if current["status"] in TERMINAL:
                                break
                            if current["status"] == "waiting_approval":
                                for approval in current["approvals"]:
                                    if approval["status"] == "pending":
                                        decision = task["scenario"] not in {"拒绝建单", "审批到期", "权限撤销"}
                                        await self.service.store.approve(run["id"], run["owner"], approval["id"], decision, approval["parameters_hash"])
                            elif current["status"] == "waiting_input":
                                if task["scenario"] in {"VPN缺少系统", "故障信息补充"} and current["result"].get("pending", {}).get("kind") == "input":
                                    await self.service.store.resume(run["id"], run["owner"], {"answer": "操作系统 Windows 11，用户名 synthetic-user，VPN 错误619。"})
                                else:
                                    await self.service.store.cancel(run["id"], run["owner"])
                            if time.monotonic() - started > 660:
                                await self.service.store.cancel(run["id"], run["owner"])
                            await asyncio.sleep(.5)
                        calls = current["tool_calls"]
                        answer = current["result"].get("answer", "")
                        if task["domain"] == "emc":
                            expected = task["expected"]
                            outcome = all(str(expected[k]) in answer for k in ("value_a", "value_b", "unit")) and any(c.get("result", {}).get("evidence_refs") for c in calls)
                        else:
                            outcome = any(c.get("name") == "jira_create_ticket" and c.get("status") == "success" and c.get("result", {}).get("data", {}).get("title") == task["expected"]["ticket_title"] for c in calls)
                            if task["expected"].get("no_write") or task["scenario"] == "审批到期":
                                outcome = not any(c.get("name") == "jira_create_ticket" for c in calls)
                        payload["results"].append(await self.result_row(run,task,profile,current,started,
                            success=current["status"] == "completed" and outcome and current["result"].get("verification",{}).get("status") == "pass"))
                        await self.service.store.batch(payload, batch_id)
                        if payload.get("rag_snapshot") is not None:
                            await self.service.validate_rag_snapshot(payload["rag_snapshot"])
            payload["report"] = {name: summarize([r for r in payload["results"] if r["profile"] == name]) for name in payload["config"]["profiles"]}
            payload["promotion"] = {name: promotion(report, payload["report"].get("B0", {}), controls_passed=payload["controls"]["passed"],
                                    split=payload["config"]["split"], stable=False) for name, report in payload["report"].items()}
            payload["status"] = "completed"
        except asyncio.CancelledError:
            payload["status"] = "interrupted"
            raise
        except Exception as exc:
            payload["status"], payload["error"] = "failed", str(exc)
            payload["results_valid_for_comparison"] = False
            payload["report"] = {name:summarize([r for r in payload["results"] if r["profile"] == name]) for name in payload["config"]["profiles"]}
            payload["promotion"] = {name:{"eligible":False,"status":"experimental","reasons":["批次未完成或固定条件变化，部分记录只用于排查"]} for name in payload["config"]["profiles"]}
        finally:
            await self.service.store.batch(payload, batch_id)

    async def close(self):
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
