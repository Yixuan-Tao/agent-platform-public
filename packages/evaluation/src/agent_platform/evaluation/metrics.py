"""Outcome-first reporting, Wilson intervals, and conservative promotion gates."""
import math
import statistics


def usage_totals(events):
    """Count trustworthy receipts plus conservative reservations missing a receipt."""
    actual = sum(e.get("usage", {}).get("total_tokens", 0) for e in events if e.get("usage_source") == "actual")
    estimated = sum(e.get("usage", {}).get("total_tokens", 0) for e in events if e.get("usage_source") == "estimated")
    ended = {e.get("model_call_id") for e in events if e.get("type") == "model" and e.get("phase") == "end"}
    estimated += sum(e.get("tokens_reserved", 0) for e in events if e.get("type") == "model" and e.get("phase") == "start" and e.get("model_call_id") not in ended)
    settled = {e.get("request_id") for e in events if e.get("type") == "rag_budget" and e.get("phase") in {"settled", "unknown"}}
    pending = [e["grant"] for e in events if e.get("type") == "rag_budget" and e.get("phase") == "reserved" and e["grant"]["request_id"] not in settled]
    estimated += sum(g["tokens"] for g in pending)
    return {"tokens_actual":actual, "tokens_estimated":estimated, "unsettled_rag_grants":len(pending)}

def wilson(successes, n):
    if not n:
        return [0., 1.]
    z, p = 1.96, successes/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [center-radius, center+radius]

def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, math.ceil(fraction*len(ordered))-1)]

def summarize(rows):
    n = len(rows)
    successes = sum(row.get("success") is True for row in rows)
    failures = {}
    for row in rows:
        if not row.get("success"):
            kind = row.get("failure_category", "unknown")
            failures[kind] = failures.get(kind, 0)+1
    return {"n": n, "successes": successes, "success_rate": successes/n if n else None,
            "confidence_interval": wilson(successes, n),
            "latency_p50": percentile([r["latency"] for r in rows], .5),
            "latency_p95": percentile([r["latency"] for r in rows], .95),
            "tokens_actual": sum(r.get("tokens_actual", 0) for r in rows),
            "tokens_estimated": sum(r.get("tokens_estimated", 0) for r in rows),
            "unsettled_rag_grants": sum(r.get("unsettled_rag_grants", 0) for r in rows),
            "cost": None, "failures": failures,
            "model_requests": sum(r.get("model_requests", 0) for r in rows),
            "tool_attempts": sum(r.get("tool_attempts", 0) for r in rows)}

def promotion(candidate, baseline, *, controls_passed, critical_errors=0, split="holdout", stable=False):
    reasons = []
    if not controls_passed:
        reasons.append("固定治理测试未全部通过")
    if split != "holdout" or not stable or candidate.get("n", 0) < 36:
        reasons.append("留出业务样本不足或差异尚不稳定")
    if (candidate.get("success_rate") or 0) < .9:
        reasons.append("留出成功率未达到90%")
    if critical_errors:
        reasons.append("存在关键证据、单位或动作错误")
    delta = (candidate.get("success_rate") or 0) - (baseline.get("success_rate") or 0)
    candidate_tokens = candidate.get("tokens_actual", 0) + candidate.get("tokens_estimated", 0)
    base_tokens = baseline.get("tokens_actual", 0) + baseline.get("tokens_estimated", 0)
    if not (delta >= .05-1e-9 or (delta >= -.02-1e-9 and base_tokens > 0 and candidate_tokens <= .8 * base_tokens)):
        reasons.append("质量提升或节省token门槛未达到")
    if baseline.get("latency_p95") is None or candidate.get("latency_p95") is None or candidate["latency_p95"] > baseline["latency_p95"] * 1.1:
        reasons.append("简单任务P95时延门槛未验证")
    return {"eligible": not reasons, "status": "stable" if not reasons else "experimental", "reasons": reasons}
