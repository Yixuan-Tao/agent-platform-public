"""
RAG 系统指标统一采集脚本

从现有数据源中采集 12 项指标，输出结构化报告并保存快照。
零侵入设计：不修改任何现有代码文件。

用法:
    python scripts/collect_metrics.py

输出:
    - 控制台打印指标报告
    - metrics/metrics_snapshot.json（快照文件，用于历史对比）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 将项目根目录加入 sys.path，确保可以导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================
# 常量
# ============================================================

SNAPSHOT_DIR = _PROJECT_ROOT / "metrics"
SNAPSHOT_FILE = SNAPSHOT_DIR / "metrics_snapshot.json"
EVAL_DIR = _PROJECT_ROOT / "evaluation"


def _latest_eval_file() -> Path:
    """按优先级选择最新的评测结果文件（v6 > v5 校准 > v5 判定 > v5）。"""
    for name in (
        "run_result_v6_judged.json", "run_result_v6.json",
        "run_result_v5_judged_calibrated.json", "run_result_v5_judged.json",
        "run_result_v5.json",
    ):
        candidate = EVAL_DIR / name
        if candidate.exists():
            return candidate
    return EVAL_DIR / "run_result_v5.json"


QA_EVAL_FILE = _latest_eval_file()

# EMC 标准文件名识别规则：(厂商关键字, 厂商名)
EMC_VENDOR_RULES = []
# 额外的 EMC 标准关键词
EMC_STANDARD_KEYWORDS = ['EMC', '电磁兼容', 'electromagnetic compatibility']


# ============================================================
# 工具函数
# ============================================================

def compute_p95(values: list[float]) -> float:
    """计算 P95 分位值。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = math.ceil(len(sorted_vals) * 0.95) - 1
    return sorted_vals[max(0, idx)]


def compute_p99(values: list[float]) -> float:
    """计算 P99 分位值。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = math.ceil(len(sorted_vals) * 0.99) - 1
    return sorted_vals[max(0, idx)]


# ============================================================
# 采集函数
# ============================================================

def collect_tool_plugin_count() -> dict[str, Any]:
    """指标1: 工具插件数量 — 从代码文件静态分析。"""
    result: dict[str, Any] = {
        "total": 0,
        "details": {},
        "sources": {},
    }

    # 1) mcp/tools.py
    mcp_tools_path = _PROJECT_ROOT / "mcp" / "tools.py"
    if mcp_tools_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("mcp_tools", mcp_tools_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mcp_tools = getattr(mod, "MCP_TOOLS", [])
                all_tools = getattr(mod, "ALL_TOOLS", [])
                result["details"]["MCP_TOOLS"] = len(mcp_tools)
                result["details"]["ALL_TOOLS"] = len(all_tools)
                # 记录工具名
                result["sources"]["mcp_tools_names"] = [
                    t.get("name", "?") for t in mcp_tools
                ]
                result["sources"]["all_tools_names"] = [
                    t.get("function", {}).get("name", "?")
                    if isinstance(t, dict) and "function" in t
                    else t.get("name", "?")
                    for t in all_tools
                ]
        except Exception as e:
            result["details"]["error"] = str(e)

    # 2) api/tool_definitions.py
    api_tools_path = _PROJECT_ROOT / "api" / "tool_definitions.py"
    if api_tools_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("api_tools", api_tools_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                tools_list = getattr(mod, "TOOLS", [])
                result["details"]["API_TOOLS"] = len(tools_list)
                result["sources"]["api_tools_names"] = [
                    t.get("function", {}).get("name", "?")
                    for t in tools_list
                ]
        except Exception as e:
            result["details"]["api_error"] = str(e)

    # 3) mcp/handlers.py
    handlers_path = _PROJECT_ROOT / "mcp" / "handlers.py"
    if handlers_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("mcp_handlers", handlers_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                handlers = getattr(mod, "HANDLERS", {})
                result["details"]["MCP_HANDLERS"] = len(handlers)
                result["sources"]["handlers_names"] = list(handlers.keys())
        except Exception as e:
            result["details"]["handlers_error"] = str(e)

    # 去重汇总：基于工具名去重（不同格式的工具可能指向同一功能）
    all_names: set[str] = set()
    for name_list in result["sources"].values():
        if isinstance(name_list, list):
            for n in name_list:
                all_names.add(str(n))
    result["total"] = len(all_names)
    result["unique_names"] = sorted(all_names)
    return result


def collect_rag_counts() -> dict[str, Any]:
    """指标5+6+7: RAG 文档数、chunk 数、图片数 — 从 ChromaDB 查询。

    D7 图片数口径：document_chunks 中 modality=image 的 chunk 数
    （即 VL 识图记录数，无论是否已生成真实图片向量）。
    image_vector_count：image_chunks 集合中真实图片向量条数，
    由百炼原生多模态向量回填后应与 image_count 相等。
    """
    result: dict[str, Any] = {
        "document_count": 0,
        "chunk_count": 0,
        "image_count": 0,
        "image_vector_count": 0,
        "documents": [],
        "error": None,
    }
    try:
        from db.chroma_client import ChromaClientSingleton
        from db.schemas import IMAGE_COLLECTION_NAME, COLLECTION_NAME

        chroma = ChromaClientSingleton.get_instance()

        # 文档数
        doc_list = chroma.list_documents()
        result["document_count"] = len(doc_list)
        result["documents"] = doc_list

        # D6 chunk 数：document_chunks 集合条数（主检索索引）；
        # 图片真实向量单独计入 image_chunks，避免 D7 图片被重复计入 chunk 数
        doc_coll = getattr(chroma, "_doc_collection", None)
        if doc_coll is not None:
            result["chunk_count"] = doc_coll.count()
        else:
            result["chunk_count"] = 0
        try:
            stats = chroma.get_collection_stats()
            result["stored_vector_total"] = stats.get("row_count", 0)
        except Exception as e:
            result["stored_vector_total_error"] = str(e)

        # D7 图片数：document_chunks 中 modality=image 的 chunk 数
        try:
            if doc_coll is not None:
                img_rows = doc_coll.get(
                    where={"modality": "image"},
                    include=["metadatas"],
                )
                result["image_count"] = len(img_rows.get("ids", []) or [])
            else:
                result["image_count"] = 0
        except Exception as e:
            result["image_count_error"] = str(e)

        # 补充：image_chunks 中真实图片向量条数（回填状态）
        try:
            if chroma._image_collection is not None:
                result["image_vector_count"] = chroma._image_collection.count()
        except Exception as e:
            result["image_vector_count_error"] = str(e)

    except Exception as e:
        result["error"] = f"ChromaDB 不可用: {e}"
    return result


def collect_answer_p95_latency() -> dict[str, Any]:
    """指标3: 单次回答 P95 时延 — 从评估结果 JSON（默认 qa_eval_full.json）计算。"""
    result: dict[str, Any] = {
        "p95_latency_ms": 0.0,
        "record_count": 0,
        "source": QA_EVAL_FILE.name,
        "error": None,
        "all_latencies": [],
    }
    if not QA_EVAL_FILE.exists():
        result["error"] = f"文件不存在: {QA_EVAL_FILE}"
        return result

    try:
        with open(QA_EVAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"读取失败: {e}"
        return result

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        result["error"] = "JSON 格式不是列表（期望 run_result 的 results 列表）"
        return result

    latencies = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # 兼容 latency_ms（intern_rag qa_eval）与 latency_s（平台 run_result）
        if item.get("latency_ms") is not None:
            latencies.append(float(item["latency_ms"]))
        elif item.get("latency_s") is not None:
            latencies.append(float(item["latency_s"]) * 1000.0)
    result["record_count"] = len(latencies)
    result["all_latencies"] = latencies

    if latencies:
        result["p95_latency_ms"] = compute_p95(latencies)
    else:
        result["error"] = "无有效的 latency_ms 数据"
    return result


def collect_eval_status_summary() -> dict[str, Any]:
    """评测新增指标：答案状态准确率、证据引用率、替换规则通过率、网关降级率。

    数据源：evaluation/run_result_v6.json（含 expected_answer_status / answer_status）。
    """
    result: dict[str, Any] = {
        "answer_status_accuracy": 0.0,
        "evidence_citation_rate": 0.0,
        "substitution_pass_rate": 0.0,
        "gateway_fallback_rate": 0.0,
        "total": 0,
        "source": str(QA_EVAL_FILE.name),
        "error": None,
    }
    if not QA_EVAL_FILE.exists():
        result["error"] = f"文件不存在: {QA_EVAL_FILE}"
        return result
    try:
        data = json.loads(QA_EVAL_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        result["error"] = f"读取失败: {exc}"
        return result
    results = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(results, list):
        result["error"] = "JSON 结构不含 results 列表"
        return result

    status_ok = status_total = 0
    evidence_ok = evidence_total = 0
    substitution_ok = substitution_total = 0
    gateway_ok = gateway_total = 0
    for item in results:
        expected = item.get("expected_answer_status")
        actual = item.get("answer_status")
        feature = str(item.get("feature") or "")
        if expected:
            # 网关故障为故障注入场景：常规运行 actual=n/a，不计入状态准确率
            if expected != "gateway_unavailable" and actual != "n/a":
                status_total += 1
                status_ok += 1 if (actual or "ok") == expected else 0
        min_evidence = item.get("expected_evidence_min") or 0
        if min_evidence:
            evidence_total += 1
            citations = item.get("citations") or item.get("contexts") or []
            evidence_ok += 1 if len(citations) >= min_evidence else 0
        if "substitution" in feature or "replace" in str(item.get("category") or ""):
            substitution_total += 1
            substitution_ok += 1 if item.get("status") == "OK" else 0
        if expected == "gateway_unavailable":
            if actual != "n/a":
                gateway_total += 1
                gateway_ok += 1 if actual == "gateway_unavailable" else 0

    result["total"] = len(results)
    result["answer_status_accuracy"] = round(status_ok / status_total, 3) if status_total else 0.0
    result["evidence_citation_rate"] = round(evidence_ok / evidence_total, 3) if evidence_total else 0.0
    result["substitution_pass_rate"] = round(substitution_ok / substitution_total, 3) if substitution_total else 0.0
    result["gateway_fallback_rate"] = round(gateway_ok / gateway_total, 3) if gateway_total else 0.0
    return result


def collect_emc_standards() -> dict[str, Any]:
    """指标11+12: EMC 标准厂商数量和标准数量 — 从 ChromaDB 文档名匹配。"""
    result: dict[str, Any] = {
        "vendor_count": 0,
        "standard_count": 0,
        "vendors": [],
        "standards": [],
        "error": None,
    }
    try:
        from db.chroma_client import ChromaClientSingleton

        chroma = ChromaClientSingleton.get_instance()
        docs = chroma.list_documents()
    except Exception as e:
        result["error"] = f"ChromaDB 不可用: {e}"
        return result

    # 从文件名匹配 EMC 标准
    emc_files: list[str] = []
    vendors_found: set[str] = set()

    for doc in docs:
        fname = doc.get("file_name", "")
        if not fname:
            continue
        # 检查是否包含 EMC 标准关键词
        is_emc = False
        for kw in EMC_STANDARD_KEYWORDS:
            if kw.lower() in fname.lower():
                is_emc = True
                break
        if not is_emc:
            # 补充：按厂商规则匹配
            for keyword, _ in EMC_VENDOR_RULES:
                if keyword.lower() in fname.lower():
                    is_emc = True
                    break
        if is_emc:
            emc_files.append(fname)
            # 识别厂商
            matched_vendor = None
            for keyword, vendor_name in EMC_VENDOR_RULES:
                if keyword.lower() in fname.lower():
                    matched_vendor = vendor_name
                    break
            if matched_vendor:
                vendors_found.add(matched_vendor)

    result["standard_count"] = len(emc_files)
    result["vendor_count"] = len(vendors_found)
    result["vendors"] = sorted(vendors_found)
    result["standards"] = sorted(emc_files)
    return result


def collect_placeholder_metrics() -> dict[str, Any]:
    """返回待积累指标的占位状态。"""
    return {
        "multi_turn_conversations": {
            "status": "pending",
            "message": "需在API层记录会话日志后采集（按session_id或时间窗口统计对话轮数）",
        },
        "tool_call_success_rate": {
            "status": "pending",
            "message": "需在工具调用点记录成功/失败次数后采集",
        },
        "retrieval_p99_latency_ms": {
            "status": "pending",
            "message": "需在Retriever.retrieve()添加计时埋点后采集",
        },
        "faithfulness_improvement_rate": {
            "status": "pending",
            "message": "需运行两次 eval 获取基线数据后对比计算",
        },
        "emc_compare_time_reduction": {
            "status": "pending",
            "message": "需在EmcCompareTool.compare()添加计时埋点后采集",
        },
    }


def collect_failure_labels() -> dict[str, Any]:
    """失败标签聚合：统计评测结果中的 failure_label 分布。"""
    result: dict[str, Any] = {
        "total": 0,
        "labels": {},
        "source": QA_EVAL_FILE.name,
        "error": None,
    }
    if not QA_EVAL_FILE.exists():
        result["error"] = f"文件不存在: {QA_EVAL_FILE}"
        return result
    try:
        with open(QA_EVAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"读取失败: {e}"
        return result
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        result["error"] = "JSON 格式不是列表（期望 results 列表）"
        return result
    try:
        from evaluation.labels import classify_failure
    except Exception:  # noqa: BLE001
        classify_failure = None
    counts: dict[str, int] = {}
    for r in results:
        label = r.get("failure_label")
        if not label and classify_failure is not None:
            label = classify_failure(r, r)  # 旧文件用结果字段兜底推导
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    result["total"] = sum(counts.values())
    result["labels"] = counts
    return result


def compare_judge_snapshots_files(prev_path: Path, cur_path: Path) -> dict[str, Any]:
    """用 Welch's t 检验对比两轮裁判快照，输出逐维度回归/改进/噪音/无变化。"""
    try:
        from evaluation.stats import compare_judge_snapshots
    except Exception as exc:  # noqa: BLE001
        return {"error": f"stats 导入失败: {exc}"}
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    return compare_judge_snapshots(prev, cur)


# ============================================================
# 快照读写
# ============================================================

def load_snapshot() -> dict[str, Any]:
    """读取历史快照，不存在时返回空。"""
    if SNAPSHOT_FILE.exists():
        try:
            with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": "1.0", "snapshots": []}


def save_snapshot(current: dict[str, Any]) -> dict[str, Any]:
    """保存当前快照（追加到历史记录中）。"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_data = load_snapshot()
    snapshots = snapshot_data.get("snapshots", [])

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": current,
    }
    snapshots.append(entry)
    # 只保留最近 10 次快照
    if len(snapshots) > 10:
        snapshots = snapshots[-10:]

    snapshot_data["snapshots"] = snapshots
    snapshot_data["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
    return snapshot_data


# ============================================================
# 报告格式化
# ============================================================

def format_report(
    tool_count: dict,
    rag_counts: dict,
    p95_latency: dict,
    emc_standards: dict,
    eval_status: dict,
    placeholders: dict,
) -> str:
    """格式化为可读的指标报告文本。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 55,
        "  RAG 系统指标报告",
        "=" * 55,
        f"  采集时间: {now}",
        "",
        "[可直接采集的指标]",
        "-" * 40,
    ]

    # 1. 工具插件数量
    tc = tool_count
    tc_detail = tc.get("details", {})
    tc_detail_str = ", ".join(
        f"{k}={v}" for k, v in tc_detail.items() if v
    )
    if tc.get("unique_names"):
        tc_detail_str += f", 去重={tc['total']}"
    lines.append(f"  1. 工具插件数量:      {tc.get('total', 0):>5} 个")
    lines.append(f"     ({tc_detail_str})")

    # 5. RAG 文档数
    rc = rag_counts
    if rc.get("error"):
        lines.append(f"  5. RAG 文档数:        [ChromaDB 不可用: {rc['error']}]")
    else:
        lines.append(f"  5. RAG 文档数:        {rc.get('document_count', 0):>5} 个")

    # 6. RAG Chunk 数
    if rc.get("error"):
        lines.append(f"  6. RAG Chunk 数:      [ChromaDB 不可用]")
    else:
        chunk_line = f"  6. RAG Chunk 数:      {rc.get('chunk_count', 0):>5} 个"
        vec_total = rc.get('stored_vector_total', 0)
        if vec_total:
            chunk_line += f"  （含图向量共 {vec_total} 条）"
        lines.append(chunk_line)

    # 7. RAG 图片数
    if rc.get("error"):
        lines.append(f"  7. RAG 图片数:       [ChromaDB 不可用]")
    else:
        img_cnt = rc.get('image_count', 0)
        vec_cnt = rc.get('image_vector_count', 0)
        line = f"  7. RAG 图片数:        {img_cnt:>5} 个"
        if vec_cnt:
            line += f"  （真实图向量 {vec_cnt} 条）"
        lines.append(line)

    # 11. EMC 厂商数量
    es = emc_standards
    if es.get("error"):
        lines.append(f" 11. EMC 厂商数量:     [ChromaDB 不可用]")
    else:
        lines.append(f" 11. EMC 厂商数量:      {es.get('vendor_count', 0):>5} 家")
        if es.get("vendors"):
            lines.append(f"     (厂商: {', '.join(es['vendors'])})")

    # 12. 标准数量
    if es.get("error"):
        lines.append(f" 12. EMC 标准数量:      [ChromaDB 不可用]")
    else:
        lines.append(f" 12. EMC 标准数量:      {es.get('standard_count', 0):>5} 份")

    # 3. 单次回答 P95 时延
    pl = p95_latency
    lines.append("")
    lines.append("[评估数据指标]")
    lines.append("-" * 40)
    if pl.get("error"):
        lines.append(f"  3. 单次回答 P95 时延:  [无数据: {pl['error']}]")
    else:
        lines.append(f"  3. 单次回答 P95 时延:  {pl.get('p95_latency_ms', 0):>8.0f} ms")
        lines.append(f"     (来源: {pl['source']}, 共 {pl.get('record_count', 0)} 条记录)")

    # 评测新增指标（答案状态 / 证据引用 / 替换规则 / 网关降级）
    es = eval_status
    lines.append("")
    lines.append("[评测证据维度指标]")
    lines.append("-" * 40)
    if es.get("error"):
        lines.append(f"  [数据缺失] {es['error']}")
    else:
        lines.append(f"  答案状态准确率:  {es.get('answer_status_accuracy', 0) * 100:.1f}%")
        lines.append(f"  证据引用率:      {es.get('evidence_citation_rate', 0) * 100:.1f}%")
        lines.append(f"  替换规则通过率:  {es.get('substitution_pass_rate', 0) * 100:.1f}%")
        lines.append(f"  网关降级率:      {es.get('gateway_fallback_rate', 0) * 100:.1f}%")
        lines.append(f"  (来源: {es['source']}, 共 {es.get('total', 0)} 条)")

    # 待积累指标
    lines.append("")
    lines.append("[待积累指标（需运行时埋点后采集）]")
    lines.append("-" * 40)
    for key, info in placeholders.items():
        label_map = {
            "multi_turn_conversations": "  2. 多轮对话轮数",
            "tool_call_success_rate": "  4. 工具调用成功率",
            "retrieval_p99_latency_ms": "  8. 检索 P99 时延",
            "faithfulness_improvement_rate": "  9. Faithfulness 提升率",
            "emc_compare_time_reduction": " 10. EMC 对比耗时降低",
        }
        label = label_map.get(key, f"  {key}")
        lines.append(f"  {label}:  [待积累] {info['message']}")

    lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================

def main():
    """采集所有指标、生成报告、保存快照。"""
    parser = argparse.ArgumentParser(description="RAG 系统指标统一采集")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="评估结果 JSON 路径（默认自动选择 evaluation/run_result_v6/v5 最新文件）")
    parser.add_argument("--compare", type=str, default=None,
                        help="与上一轮裁判快照对比（传入该快照 JSON 路径，输出 Welch's t 回归检测）")
    args = parser.parse_args()

    # 允许 CLI 指定评估结果文件（P95 时延数据源）
    global QA_EVAL_FILE
    if args.eval_file:
        QA_EVAL_FILE = _PROJECT_ROOT / args.eval_file

    print("正在采集 RAG 系统指标...")

    # 采集
    tool_count = collect_tool_plugin_count()
    rag_counts = collect_rag_counts()
    p95_latency = collect_answer_p95_latency()
    emc_standards = collect_emc_standards()
    eval_status = collect_eval_status_summary()
    failure_labels = collect_failure_labels()
    placeholders = collect_placeholder_metrics()

    # 生成报告
    report = format_report(tool_count, rag_counts, p95_latency, emc_standards, eval_status, placeholders)
    print("\n" + report)

    if failure_labels.get("labels"):
        print("\n[失败标签分布]")
        for label, count in sorted(failure_labels["labels"].items()):
            print(f"  {label}: {count}")

    judge_compare = None
    if args.compare:
        judge_compare = compare_judge_snapshots_files(Path(args.compare), QA_EVAL_FILE)
        print("\n[裁判回归检测]")
        print(json.dumps(judge_compare, ensure_ascii=False, indent=2))

    # 保存快照
    snapshot_data = {
        "tool_plugin_count": {"value": tool_count.get("total", 0), "details": tool_count.get("details", {})},
        "rag_document_count": {"value": rag_counts.get("document_count", 0)},
        "rag_chunk_count": {"value": rag_counts.get("chunk_count", 0)},
        "rag_image_count": {"value": rag_counts.get("image_count", 0)},
        "rag_image_vector_count": {"value": rag_counts.get("image_vector_count", 0)},
        "rag_stored_vector_total": {"value": rag_counts.get("stored_vector_total", 0)},
        "answer_p95_latency_ms": {"value": p95_latency.get("p95_latency_ms", 0.0),
                                   "record_count": p95_latency.get("record_count", 0)},
        "answer_status_accuracy": {"value": eval_status.get("answer_status_accuracy", 0.0)},
        "evidence_citation_rate": {"value": eval_status.get("evidence_citation_rate", 0.0)},
        "substitution_pass_rate": {"value": eval_status.get("substitution_pass_rate", 0.0)},
        "gateway_fallback_rate": {"value": eval_status.get("gateway_fallback_rate", 0.0)},
        "failure_labels": {"total": failure_labels.get("total", 0),
                            "labels": failure_labels.get("labels", {})},
        "emc_vendor_count": {"value": emc_standards.get("vendor_count", 0),
                              "vendors": emc_standards.get("vendors", [])},
        "emc_standard_count": {"value": emc_standards.get("standard_count", 0)},
    }
    if judge_compare:
        snapshot_data["judge_regression"] = judge_compare
    saved = save_snapshot(snapshot_data)
    # 检查历史是否有多条
    history_count = len(saved.get("snapshots", []))
    print(f"\n快照已保存到: {SNAPSHOT_FILE} (共 {history_count} 次快照)")
    if history_count >= 2:
        # 对比最新两次快照
        prev = saved["snapshots"][-2]
        curr = saved["snapshots"][-1]
        print("\n与上次快照对比:")
        for key in ["rag_document_count", "rag_chunk_count", "rag_image_count"]:
            prev_val = prev.get("metrics", {}).get(key, {}).get("value", 0)
            curr_val = curr.get("metrics", {}).get(key, {}).get("value", 0)
            diff = curr_val - prev_val
            sign = "+" if diff > 0 else ""
            print(f"  {key}: {prev_val} → {curr_val} ({sign}{diff})")


if __name__ == "__main__":
    main()
