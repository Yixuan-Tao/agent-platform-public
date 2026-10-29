# Test values are synthetic fixtures for software behavior.
from collections import Counter
from agent_platform.evaluation.dataset import dataset, dataset_hash
from agent_platform.evaluation.metrics import summarize, promotion, wilson
from agent_platform.evaluation.metrics import usage_totals

def test_dataset_is_stratified_versioned_and_preserves_holdout():
    tasks = dataset()
    assert len(tasks) == 80
    assert len({t["id"] for t in tasks}) == 80
    assert Counter(t["domain"] for t in tasks) == {"emc": 40, "it": 20, "governance": 20}
    assert Counter(t["split"] for t in tasks) == {"development": 48, "validation": 16, "holdout": 16}
    assert len(dataset_hash()) == 64
    assert all(t["expected"] and t["assertions"] for t in tasks)

def test_failures_stay_in_denominator_and_usage_never_becomes_price():
    rows = [{"success": True, "latency": 1, "tokens_actual": 100},
            {"success": False, "failure_category": "timeout", "latency": 600, "tokens_estimated": 200}]
    report = summarize(rows)
    assert report["n"] == 2 and report["success_rate"] == .5
    assert report["latency_p95"] == 600
    assert report["tokens_actual"] == 100 and report["tokens_estimated"] == 200
    assert report["cost"] is None
    low, high = report["confidence_interval"]
    assert low < .5 < high
    assert not promotion(report, report, controls_passed=False)["eligible"]

def test_promotion_requires_controls_holdout_stability_and_cost_or_quality():
    baseline = {"n": 36, "success_rate": .9, "tokens_actual": 1000, "latency_p95": 10}
    candidate = {"n": 36, "success_rate": .95, "tokens_actual": 900, "latency_p95": 10}
    # Floating differences at the threshold need a numerical tolerance.
    eligible = promotion(candidate, baseline, controls_passed=True, stable=True)
    assert eligible["eligible"], eligible
    assert not promotion(candidate, baseline, controls_passed=True, stable=True, critical_errors=1)["eligible"]

def test_cancelled_run_unknown_reservations_are_estimates_without_double_counting():
    events = [{"type":"model","phase":"end","model_call_id":"actual","usage_source":"actual","usage":{"total_tokens":10}},
              {"type":"model","phase":"start","model_call_id":"lost","tokens_reserved":100},
              {"type":"rag_budget","phase":"reserved","grant":{"request_id":"rag-lost","tokens":32000}}]
    assert usage_totals(events) == {"tokens_actual":10,"tokens_estimated":32100,"unsettled_rag_grants":1}
    events += [{"type":"model","phase":"end","model_call_id":"lost","usage_source":"estimated","usage":{"total_tokens":100}},
               {"type":"rag_budget","phase":"unknown","request_id":"rag-lost"},
               {"type":"model","phase":"end","model_call_id":"rag-lost","usage_source":"estimated","usage":{"total_tokens":32000}}]
    assert usage_totals(events) == {"tokens_actual":10,"tokens_estimated":32100,"unsettled_rag_grants":0}
