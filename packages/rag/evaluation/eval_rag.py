"""
RAG Agent 效果评测框架

基于 ragas 框架，提供以下评测维度：

1. 检索质量（Retrieval Quality）:
   - Context Precision: 检索到的上下文是否精准相关
   - Context Recall: 相关上下文是否被检索到
   - Context Relevancy: 上下文的整体相关性

2. 生成质量（Generation Quality）:
   - Faithfulness: 答案是否忠实于上下文（无幻觉）
   - Answer Relevancy: 答案与问题的相关性
   - Answer Correctness: 答案的正确性

3. 端到端质量（End-to-End）:
   - 答案完整性
   - 引用准确性
   - 多语言支持质量

用法:
    python -m evaluation.eval_rag --testset evaluation/testset.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ragas 是可选依赖，评测环境下安装
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
        answer_correctness,
    )
    _RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAGAS_AVAILABLE = False


# ============================================================
# 评测指标定义
# ============================================================

@dataclass
class EvalMetrics:
    """单条评测结果。"""
    question: str
    answer: str
    ground_truth: str = ""
    # 检索指标
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0
    # 生成指标
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0
    # 元信息
    latency_ms: float = 0.0
    language: str = "zh"
    error: str = ""


@dataclass
class EvalReport:
    """评测报告。"""
    total_questions: int = 0
    success_count: int = 0
    error_count: int = 0
    # 各指标平均值
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_context_relevancy: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_answer_correctness: float = 0.0
    avg_latency_ms: float = 0.0
    # 详细结果
    details: list[EvalMetrics] = field(default_factory=list)
    # 按语种分组
    by_language: dict[str, dict] = field(default_factory=dict)


# ============================================================
# 评测器
# ============================================================

class RAGEvaluator:
    """RAG 效果评测器：编排检索 → 生成 → 评分全流程。"""

    def __init__(self, query_pipeline=None):
        """
        Args:
            query_pipeline: QueryPipeline 实例，若为 None 则需手动传入答案。
        """
        self._pipeline = query_pipeline

    def evaluate_from_testset(
        self,
        testset: list[dict],
        concurrency: int = 1,
    ) -> EvalReport:
        """从测试集批量评测。

        testset 格式:
            [
                {
                    "question": "ESD 击穿电压是多少？",
                    "ground_truth": "27V",
                    "language": "zh",
                    "relevant_chunks": ["c1", "c2"],  # 可选，用于 context_recall
                },
                ...
            ]

        Returns:
            EvalReport 评测报告。
        """
        report = EvalReport(total_questions=len(testset))
        results: list[EvalMetrics] = []

        for i, item in enumerate(testset):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")
            language = item.get("language", "zh")

            t0 = time.time()
            metrics = EvalMetrics(
                question=question,
                answer="",
                ground_truth=ground_truth,
                language=language,
            )

            try:
                # 1) 执行 RAG 查询
                if self._pipeline:
                    result = self._pipeline.query(
                        question, use_rerank=True, language=language,
                    )
                    metrics.answer = result.get("answer", "")
                    contexts = result.get("sources", [])
                else:
                    # 手动评测模式：testset 中已包含 answer 和 contexts
                    metrics.answer = item.get("answer", "")
                    contexts = item.get("contexts", [])

                # 2) 计算 ragas 指标（如果可用）
                if _RAGAS_AVAILABLE and metrics.answer and contexts:
                    self._compute_ragas_metrics(metrics, question, contexts, ground_truth, item)

                metrics.latency_ms = (time.time() - t0) * 1000
                report.success_count += 1

            except Exception as e:
                metrics.error = str(e)
                metrics.latency_ms = (time.time() - t0) * 1000
                report.error_count += 1

            results.append(metrics)
            print(f"[{i + 1}/{len(testset)}] {question[:50]}... "
                  f"({metrics.latency_ms:.0f}ms)")

        # 汇总统计
        report.details = results
        self._aggregate(report)
        return report

    def _compute_ragas_metrics(
        self,
        metrics: EvalMetrics,
        question: str,
        contexts: list[dict],
        ground_truth: str,
        item: dict,
    ) -> None:
        """使用 ragas 计算各项指标。"""
        context_texts = [c.get("content", "") for c in contexts]
        if not context_texts:
            return

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [metrics.answer],
            "contexts": [context_texts],
            "ground_truth": [ground_truth],
        })

        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    context_relevancy,
                ],
            )
            df = result.to_pandas()
            if not df.empty:
                row = df.iloc[0]
                metrics.faithfulness = float(row.get("faithfulness", 0) or 0)
                metrics.answer_relevancy = float(row.get("answer_relevancy", 0) or 0)
                metrics.context_precision = float(row.get("context_precision", 0) or 0)
                metrics.context_recall = float(row.get("context_recall", 0) or 0)
                metrics.context_relevancy = float(row.get("context_relevancy", 0) or 0)
        except Exception as e:
            metrics.error = f"ragas 评测失败: {e}"

    def _aggregate(self, report: EvalReport) -> None:
        """汇总各指标平均值。"""
        success = [m for m in report.details if not m.error]
        if not success:
            return

        n = len(success)
        report.avg_context_precision = sum(m.context_precision for m in success) / n
        report.avg_context_recall = sum(m.context_recall for m in success) / n
        report.avg_context_relevancy = sum(m.context_relevancy for m in success) / n
        report.avg_faithfulness = sum(m.faithfulness for m in success) / n
        report.avg_answer_relevancy = sum(m.answer_relevancy for m in success) / n
        report.avg_answer_correctness = sum(m.answer_correctness for m in success) / n
        report.avg_latency_ms = sum(m.latency_ms for m in success) / n

        # 按语种分组
        for lang in ("zh", "en", "de"):
            lang_results = [m for m in success if m.language == lang]
            if lang_results:
                report.by_language[lang] = {
                    "count": len(lang_results),
                    "avg_faithfulness": sum(m.faithfulness for m in lang_results) / len(lang_results),
                    "avg_latency_ms": sum(m.latency_ms for m in lang_results) / len(lang_results),
                }

    def print_report(self, report: EvalReport) -> None:
        """打印格式化的评测报告。"""
        print("\n" + "=" * 60)
        print("  RAG Agent 效果评测报告")
        print("=" * 60)
        print(f"  总问题数:     {report.total_questions}")
        print(f"  成功:         {report.success_count}")
        print(f"  失败:         {report.error_count}")
        print(f"  平均延迟:     {report.avg_latency_ms:.0f} ms")
        print("-" * 60)
        print("  检索质量:")
        print(f"    Context Precision:  {report.avg_context_precision:.3f}")
        print(f"    Context Recall:     {report.avg_context_recall:.3f}")
        print(f"    Context Relevancy:  {report.avg_context_relevancy:.3f}")
        print("-" * 60)
        print("  生成质量:")
        print(f"    Faithfulness:       {report.avg_faithfulness:.3f}")
        print(f"    Answer Relevancy:   {report.avg_answer_relevancy:.3f}")
        print(f"    Answer Correctness: {report.avg_answer_correctness:.3f}")
        print("-" * 60)

        if report.by_language:
            print("  按语种分组:")
            for lang, stats in report.by_language.items():
                lang_name = {"zh": "中文", "en": "English", "de": "Deutsch"}.get(lang, lang)
                print(f"    {lang_name} ({stats['count']} 条): "
                      f"Faithfulness={stats['avg_faithfulness']:.3f}, "
                      f"Latency={stats['avg_latency_ms']:.0f}ms")
        print("=" * 60)

        # 输出错误详情
        errors = [m for m in report.details if m.error]
        if errors:
            print(f"\n  错误详情 ({len(errors)} 条):")
            for e in errors:
                print(f"    - [{e.language}] {e.question[:60]}: {e.error}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Agent 效果评测")
    parser.add_argument(
        "--testset", type=str, default="evaluation/testset_v2.json",
        help="测试集 JSON 文件路径",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/report.json",
        help="评测报告输出路径",
    )
    args = parser.parse_args()

    # 加载测试集
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"[错误] 测试集文件不存在: {testset_path}")
        print("请先运行 python -m evaluation.eval_testset 生成测试集。")
        exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    # 初始化 pipeline
    from pipeline.query import QueryPipeline
    pipeline = QueryPipeline()

    # 执行评测
    evaluator = RAGEvaluator(query_pipeline=pipeline)
    report = evaluator.evaluate_from_testset(testset)
    evaluator.print_report(report)

    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": report.total_questions,
            "success": report.success_count,
            "errors": report.error_count,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_answer_relevancy": report.avg_answer_relevancy,
            "avg_context_precision": report.avg_context_precision,
            "avg_context_recall": report.avg_context_recall,
            "by_language": report.by_language,
            "details": [
                {
                    "question": m.question,
                    "answer": m.answer[:200],
                    "faithfulness": m.faithfulness,
                    "latency_ms": m.latency_ms,
                    "error": m.error,
                }
                for m in report.details
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n评测报告已保存到: {output_path}")
