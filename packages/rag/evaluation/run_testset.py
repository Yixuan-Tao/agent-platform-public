"""按预置评测集跑端到端 RAG 评测（不依赖 ragas，用 expected_keywords 命中率打分）。

用法（在 packages/rag 目录下）:
    python evaluation/run_testset.py [--testset tests/testset_v3.json] [--output evaluation/run_result_v3.json]

输出：
    - 每题：状态(OK/MISS/ERR)、耗时、来源数、类别
    - 汇总：命中率、平均耗时，按 doc_group / category / difficulty 分组
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # 减少评测过程日志噪音
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config  # noqa: E402
from pipeline.query import QueryPipeline  # noqa: E402


def judge_targets_of(results: list[dict]) -> list[dict]:
    """需要裁判评分的题目：排除执行失败与正常运行中标记为 N/A 的用例。"""
    return [r for r in results if r.get("status") not in ("ERR", "N/A")]


def to_judge_items(targets: list[dict]) -> list[dict]:
    """把结果条目转成裁判输入。"""
    return [
        {
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "contexts": r.get("contexts") or [],
            "ground_truth": r.get("ground_truth", ""),
            "expected_keywords": r.get("expected_keywords") or [],
            "judge_criteria": r.get("judge_criteria", ""),
            "doc_group": r.get("doc_group", ""),
            "expected_answer_status": r.get("expected_answer_status"),
            "expected_evidence_min": r.get("expected_evidence_min"),
            "keyword_variants": r.get("keyword_variants") or [],
        }
        for r in targets
    ]


def backfill_judge(targets: list[dict], judge_results: list[dict]) -> None:
    """把裁判结果按位置写回同一批目标；数量不足时补空字典。

    调用方必须传入与 ``to_judge_items`` 同源的 ``targets``，否则会出现
    "N/A 题吃掉后一题评分" 的错位。
    """
    for r, judged in zip(targets, list(judge_results) + [{}] * len(targets)):
        r["judge"] = judged


def _normalize(text: str) -> str:
    """归一化：小写、全角→半角、µ/μ→u、去空白与括号连字符，便于变体匹配。"""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace("µ", "u").replace("μ", "u")
    for char in (" ", "-", "_", "/", "\\", "(", ")", "（", "）", ":", "：", ".", "·"):
        text = text.replace(char, "")
    return text


_EVAL_DIR = Path(__file__).resolve().parent
_KEYWORD_VARIANTS: dict[str, list[str]] = {
    _normalize(key): [_normalize(variant) for variant in variants]
    for key, variants in json.loads(
        (_EVAL_DIR / "keyword_variants.json").read_text(encoding="utf-8")
    ).items()
    if key != "_comment"
}


def _expand_keywords(item: dict) -> list[str]:
    """展开用例关键词：原文关键词 ∪ 全局变体表 ∪ 用例级 expected_keyword_variants。"""
    expanded: list[str] = []
    for kw in item.get("expected_keywords", []):
        key = _normalize(kw)
        expanded.append(key)
        for variant in _KEYWORD_VARIANTS.get(key, []):
            expanded.append(_normalize(variant))
    for variant in item.get("expected_keyword_variants", []):
        expanded.append(_normalize(variant))
    # 去重并保持顺序
    return list(dict.fromkeys(v for v in expanded if v))


def _keyword_hit(answer: str, expanded: list[str]) -> bool:
    if not expanded:
        return bool(answer)
    norm = _normalize(answer)
    return any(k in norm for k in expanded)


def _resolve_doc_id(chroma, key: str) -> str:
    """按文件名子串解析真实 doc_id（从 document_chunks metadata 查询）。

    注意：doc_id 是摄入时生成的非空 hex 值，不能从文件名前缀推导；
    旧实现按 ``file_name.split("_", 1)[0]`` 解析会导致取不到文档，
    对比工具降级为兜底文案并被误判为下游 ok。
    """
    if chroma is None or chroma._doc_collection is None:
        return ""
    for doc in chroma.list_documents():
        fname = str(doc.get("file_name", "") or "")
        if not fname or key.lower() not in fname.lower():
            continue
        try:
            rows = chroma._doc_collection.get(
                where={"file_name": fname},
                include=["metadatas"],
                limit=1,
            )
            metadatas = rows.get("metadatas") or []
            if metadatas and metadatas[0]:
                return str(metadatas[0].get("doc_id", ""))
        except Exception:  # noqa: BLE001
            pass
    return ""


def _run_downstream_confirm(pipeline: QueryPipeline, item: dict, query: str) -> str:
    """路由到 emc_compare 后执行真实对比，确认下游有证据结论。"""
    docs = item.get("downstream_docs") or {}
    if not docs.get("a") or not docs.get("b"):
        return "error:missing_docs"
    try:
        from db.chroma_client import ChromaClientSingleton
        from tools.emc_compare import EmcCompareTool
        from config import config

        chroma = ChromaClientSingleton.get_instance()
        doc_a = _resolve_doc_id(chroma, docs["a"])
        doc_b = _resolve_doc_id(chroma, docs["b"])
        if not doc_a or not doc_b:
            return "error:doc_not_found"
        tool = EmcCompareTool(
            rag_service=pipeline.rag_service,
            generator=pipeline.generator,
            config=config,
        )
        result = tool.compare(
            query=query, doc_a=doc_a, doc_b=doc_b,
            language=item.get("language", "zh"),
        )
        sources = result.get("sources") or []
        answer = result.get("answer") or ""
        return "ok" if (sources or answer) else "insufficient_evidence"
    except Exception as exc:  # noqa: BLE001
        return f"error:{str(exc)[:120]}"


_AGENT_PROMPTS = {
    "it_support": (
        "你是企业 IT 支持专家，负责 FAQ 解答、故障（Incident）处理、服务请求（Service Request）"
        "受理与工单创建。请输出结构化、可直接执行的处理方案："
        "1) 故障类型判定（FAQ/Incident/Service Request）；"
        "2) 排查/处理步骤（分步列出）；"
        "3) 需要采集的关键信息清单（如账号、报错、时间、影响范围）；"
        "4) 是否创建 Jira 工单及工单关键字段；"
        "5) 何时升级人工（escalate）。"
    ),
    "info_collect": (
        "你是行业信息采集与洞察 Agent，负责网页/RSS 采集、过滤去噪、提炼洞察并生成结构化报告。"
        "请输出结构化方案：1) 数据来源与抓取方式（网页/RSS、关键词）；"
        "2) 过滤与去噪规则；3) 拟提炼的洞察维度；4) 报告模板结构（主题/来源/时间范围/关键发现/结论建议）。"
    ),
    "orchestrator": (
        "你是多 Agent 平台的编排路由器。请判断用户请求应路由给哪个 Agent"
        "（it_support / hardware_qa / emc_compare / info_collect）。输出 JSON："
        '{"agent": "<agent_name>", "reason": "<一句话理由>", "confidence": <0-1>}。'
        "仅当请求与平台业务完全无关（如天气、闲聊、与 IT/硬件/EMC/行业信息均无关）时，"
        "必须拒绝并输出"
        '{"agent": "unsupported", "reason": "超出平台能力，建议转人工或外部服务", "confidence": 1.0}，'
        "不得强行路由到任一 Agent；行业动态/资讯采集属 info_collect 能力范围，应正常路由。"
        "当路由到 emc_compare 时，除 JSON 外还必须给出完整的对比执行方案（用要点列出）："
        "1) 分别提取两份标准的全部试验项目清单（含参数/限值/判定）；"
        "2) 按试验项逐项对比两边的频率范围、等级、限值、判定要求；"
        "3) 汇总主要差异并标注来源。方案要具体到可执行，不能只有一句话。"
    ),
    "platform": (
        "你是统一企业 Agent 平台（agent-platform）的架构工程师。请基于平台真实设计正面解释机制，用要点列出目的与工作方式，不要泛泛而谈。平台机制事实："
        "1) Agent 注册：AgentManifest 声明式注册进 AgentRegistry（name/description/capabilities/tools/handoff），构建器注册后自动进入路由表，接入层无需改代码；"
        "2) 路由与编排：SemanticRouter 关键词路由 + LLM 兜底；LangGraph Supervisor 多 Agent 编排（ROUTE→SUPERVISOR→AGENT_EXEC→CHECKER），支持 Router-as-ReAct（AgentTool 以工具方式调度子 Agent）与 Plan-and-Execute；"
        "3) 反思校验：Checker 节点检查回答完整性与幻觉，ReflectionLoop 最多重做 2 轮；"
        "4) 会话与记忆：短期记忆 ShortTermMemory 按 session_id 保留最近 10 条消息（Redis，TTL 1h）；长期记忆 LongTermMemory 把被点赞的问答对持久化到 ChromaDB（feedback_qa_pairs），后续会话检索相似 QA（score≥0.5、最多 3 条、≤500 token）作为 few-shot 注入；"
        "5) 反馈闭环：FeedbackService 收到好评延迟约 5 秒入库（窗口内可撤销 undo），差评/修正记录原因；"
        "6) 工具与沙箱：统一 BaseTool 契约 + ToolRegistry 注册表；MCP 服务端可暴露工具、MCP 客户端可接入外部服务器；用户插件经 PluginTool 在子进程沙箱（JSON IPC）中隔离执行；"
        "7) 网关与安全：模型调用统一经 LLMRouter（重试/熔断），支持用户认证、审计与 TLS。"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="tests/testset_v3.json")
    parser.add_argument("--output", default="evaluation/run_result_v3.json")
    parser.add_argument("--judge", action="store_true",
                        help="用 DeepSeek LLM-as-Judge 对每条回答打分（需 DEEPSEEK_API_KEY）")
    parser.add_argument("--judge-output", default="evaluation/run_result_judged.json")
    parser.add_argument("--judge-from",
                        help="对已保存的评测结果文件（含 answer/contexts）直接跑裁判，跳过 RAG 回答")
    args = parser.parse_args()

    if args.judge_from:
        saved = json.loads(Path(args.judge_from).read_text(encoding="utf-8"))
        results = saved["results"]
        summary = saved.get("summary", {})
        print(f"加载 {len(results)} 条已保存结果，开始裁判 ...")
    else:
        results = _run_rag(args.testset)
        summary = _keyword_summary(results)
        print("\n==== 关键词汇总 ====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    judge_summary = None
    if args.judge or args.judge_from:
        from evaluation.judge import DeepSeekJudge, summarize_judge

        judge = DeepSeekJudge()
        if not judge.available:
            print("\n[裁判] 未设置 DEEPSEEK_API_KEY，跳过 LLM 评判（仅输出关键词命中率）")
            judge_summary = {"judged": 0, "note": "缺少 DEEPSEEK_API_KEY"}
        else:
            print(f"\n[裁判] 使用 {judge.model}（{judge.base_url}）评判 {len(results)} 条回答 ...")
            # 裁判对象与回填对象必须用同一个过滤条件，否则 N/A 题会错位占用
            # 后一题的评分（历史 bug）。
            judge_targets = judge_targets_of(results)
            judge_items = to_judge_items(judge_targets)
            judge_results = judge.judge_many(judge_items)
            backfill_judge(judge_targets, judge_results)
            judge_summary = summarize_judge(judge_results)
            print("\n==== 裁判汇总 ====")
            print(json.dumps(judge_summary, ensure_ascii=False, indent=2))

    out_path = args.judge_from or args.output
    if args.judge or args.judge_from:
        out_path = args.judge_output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_data = {"summary": summary, "results": results}
    if judge_summary:
        out_data["judge_summary"] = judge_summary
    Path(out_path).write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n结果已写入:", out_path)
    return 0


def _run_rag(testset_path: str) -> list[dict]:
    """执行 RAG 问答，返回逐题结果。"""
    testset = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    pipeline = QueryPipeline()
    results: list[dict] = []

    for i, item in enumerate(testset):
        q = item.get("question", "")
        kws = [str(k).lower() for k in item.get("expected_keywords", [])]
        expanded_kws = _expand_keywords(item)
        agent = item.get("agent", "rag")
        agent_prompt = _AGENT_PROMPTS.get(agent)
        kb_top_k = int(item.get("top_k") or 8)
        res: dict = {}
        t0 = time.time()
        try:
            if agent_prompt:
                # Agent 场景题：用对应角色提示词直接生成（不依赖知识库）
                generator = pipeline.generator
                ans = generator.generate(
                    q, [], language=item.get("language", "zh"), system_prompt=agent_prompt,
                )
                sources = []
            else:
                res = pipeline.query(
                    q, top_k=kb_top_k, use_rerank=True, language=item.get("language", "zh"),
                )
                ans = (res.get("answer") or "").strip()
                sources = res.get("sources") or []
                # 知识题召回为空时重试一次（容忍偶发的嵌入请求失败）
                if not sources and item.get("doc_group") != "none":
                    res = pipeline.query(
                        q, top_k=kb_top_k, use_rerank=True, language=item.get("language", "zh"),
                    )
                    ans = (res.get("answer") or "").strip()
                    sources = res.get("sources") or []
            hit = _keyword_hit(ans, expanded_kws) if expanded_kws else None
            status = "OK" if hit else "MISS"

            # ---- orchestrator 路由用例：下游 emc_compare 结果确认 ----
            downstream_status = None
            if item.get("downstream_confirm") == "emc_compare":
                downstream_status = _run_downstream_confirm(pipeline, item, q)
                if downstream_status != "ok":
                    status = "MISS" if status == "OK" else status
                    hit = False

            # ---- 证据/状态维度判定 ----
            expected_status = item.get("expected_answer_status")
            feature = str(item.get("feature") or "")
            actual_status = "ok"
            if expected_status == "insufficient_evidence":
                no_data_markers = (
                    "未提及", "未提供", "无数据", "未找到", "无法提供", "无法给出",
                    "并未包含", "不包含", "没有该", "没有找到", "并未提供",
                    "无法直接替换", "不能直接替换", "证据不足", "无法确定",
                    "无法判断", "不能确定", "不确定", "无法确认",
                )
                actual_status = "insufficient_evidence" if (
                    not ans or any(marker in ans for marker in no_data_markers)
                ) else "ok"
            elif expected_status == "gateway_unavailable":
                actual_status = "gateway_unavailable" if "Gateway" in str(res.get("error", "")) else "n/a"
            if feature == "rag.stream_consistency":
                try:
                    stream_text = "".join(pipeline.query_stream(
                        q, top_k=kb_top_k, use_rerank=True, language=item.get("language", "zh"),
                    ))
                except Exception:  # noqa: BLE001
                    stream_text = ""
                stream_hit = bool(stream_text) and (
                    _keyword_hit(stream_text, expanded_kws) if expanded_kws else True
                )
                actual_status = "ok" if stream_hit else "insufficient_evidence"
            status_hit = (
                (actual_status == expected_status)
                if expected_status and expected_status != "gateway_unavailable" else None
            )
            # 有答案状态判定的用例以状态为准；网关失败用例在常规运行中标记 N/A
            if expected_status == "gateway_unavailable":
                status = "N/A"
            elif status_hit is not None:
                status = "OK" if status_hit else "MISS"
                hit = status_hit
            min_evidence = item.get("expected_evidence_min") or 0
            evidence_ok = len(sources) >= min_evidence if min_evidence else None
            failure_label = None
            if status not in ("OK", "N/A"):
                from evaluation.labels import classify_failure

                failure_label = classify_failure(item, {
                    "status": status,
                    "answer_status": actual_status,
                    "evidence_ok": evidence_ok,
                    "sources": len(sources),
                    "keyword_hit": bool(hit),
                })

            results.append({
                "question": q,
                "feature": feature,
                "category": item.get("category"),
                "doc_group": item.get("doc_group"),
                "difficulty": item.get("difficulty"),
                "status": status,
                "keyword_hit": bool(hit),
                "keyword_variants": expanded_kws,
                "downstream_status": downstream_status,
                "expected_answer_status": expected_status,
                "answer_status": actual_status,
                "status_hit": status_hit,
                "expected_evidence_min": min_evidence,
                "evidence_ok": evidence_ok,
                "failure_label": failure_label,
                "latency_s": round(time.time() - t0, 2),
                "sources": len(sources),
                "contexts": [
                    (s.get("content") or "")[:1200]
                    for s in sources[: config.JUDGE_CONTEXTS_MAX]
                ],
                # 完整回答入库（裁判/审计需要全文，不允许截断为部分回答）
                "answer": ans,
                "expected_keywords": kws,
                "ground_truth": item.get("ground_truth", ""),
                "judge_criteria": item.get("judge_criteria", ""),
            })
            print(f"[{i + 1}/{len(testset)}] {status} {str(item.get('category'))[:10]:10s} "
                  f"{time.time() - t0:5.1f}s src={len(sources)} | {q[:42]}")
        except Exception as e:  # noqa: BLE001
            results.append({
                "question": q,
                "category": item.get("category"),
                "doc_group": item.get("doc_group"),
                "difficulty": item.get("difficulty"),
                "status": "ERR",
                "keyword_hit": False,
                "failure_label": "FORMAT_FAIL",
                "latency_s": round(time.time() - t0, 2),
                "error": str(e)[:160],
            })
            print(f"[{i + 1}/{len(testset)}] ERR {time.time() - t0:5.1f}s | {q[:42]} | {str(e)[:70]}")
        sys.stdout.flush()
    return results


def _keyword_summary(results: list[dict]) -> dict:
    """关键词命中率汇总 + 证据/状态维度统计（兼容旧输出）。"""
    total = len(results)
    hits = sum(1 for r in results if r.get("keyword_hit"))
    errs = sum(1 for r in results if r.get("status") == "ERR")
    lats = [r["latency_s"] for r in results if r.get("status") != "ERR"]

    summary = {
        "total": total,
        "keyword_hit": hits,
        "hit_rate": round(hits / total, 3) if total else 0.0,
        "errors": errs,
        "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else 0.0,
        "answer_status_accuracy": 0.0,
        "evidence_citation_rate": 0.0,
        "substitution_pass_rate": 0.0,
        "gateway_cases": 0,
        "downstream_confirm": 0,
        "downstream_ok_rate": 0.0,
    }
    status_total = sum(1 for r in results if r.get("status_hit") is not None)
    status_hits = sum(1 for r in results if r.get("status_hit"))
    summary["answer_status_accuracy"] = round(status_hits / status_total, 3) if status_total else 0.0

    evidence_total = sum(1 for r in results if r.get("expected_evidence_min"))
    evidence_hits = sum(1 for r in results if r.get("evidence_ok"))
    summary["evidence_citation_rate"] = round(evidence_hits / evidence_total, 3) if evidence_total else 0.0

    substitution = [r for r in results if "substitution" in str(r.get("feature") or "")]
    summary["substitution_pass_rate"] = round(
        sum(1 for r in substitution if r.get("keyword_hit") or r.get("judge", {}).get("pass")) / len(substitution), 3,
    ) if substitution else 0.0

    summary["gateway_cases"] = sum(1 for r in results if r.get("expected_answer_status") == "gateway_unavailable")
    downstream = [r for r in results if r.get("downstream_status") is not None]
    summary["downstream_confirm"] = len(downstream)
    summary["downstream_ok_rate"] = round(
        sum(1 for r in downstream if r.get("downstream_status") == "ok") / len(downstream), 3,
    ) if downstream else 0.0
    for dim in ("doc_group", "category", "difficulty"):
        groups: dict[str, dict] = {}
        for r in results:
            key = str(r.get(dim) or "?")
            g = groups.setdefault(key, {"n": 0, "hit": 0})
            g["n"] += 1
            g["hit"] += 1 if r.get("keyword_hit") else 0
        summary[f"by_{dim}"] = {
            k: {**v, "rate": round(v["hit"] / v["n"], 3)} for k, v in groups.items()
        }
    from collections import Counter

    summary["failure_labels"] = dict(
        Counter(r.get("failure_label") for r in results if r.get("failure_label"))
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
