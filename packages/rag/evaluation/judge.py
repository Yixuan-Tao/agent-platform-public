"""DeepSeek LLM-as-Judge 评判器。

用独立的大模型（默认 DeepSeek）按评分规则（G-Eval / rubric 风格）对
RAG 回答逐题打分，输出结构化 JSON：
    {
      "score": 0-5 总分,
      "faithfulness": 0-5 忠实度（是否基于检索上下文、有无编造）,
      "relevance": 0-5 相关性（是否正面回答问题）,
      "completeness": 0-5 完整性（关键点是否齐全）,
      "citation_quality": 0-5 引用质量（是否标注来源、引用是否可信）,
      "pass": bool（score >= JUDGE_PASS_THRESHOLD）,
      "feedback": 简短改进建议
    }

配置（根 .env / 环境变量）：
    DEEPSEEK_API_KEY         必填，DeepSeek 官方 API Key
    DEEPSEEK_BASE_URL        默认 https://api.deepseek.com（OpenAI 兼容）
    DEEPSEEK_MODEL           默认 deepseek-chat（可选 deepseek-v4-flash / deepseek-v4-pro）
    DEEPSEEK_MODEL_VERSION   固定模型版本快照（留痕用，建议钉死避免上游漂移）
    DEEPSEEK_TIMEOUT         默认 60s
    JUDGE_PASS_THRESHOLD     默认 4.0
    JUDGE_MAX_WORKERS        默认 4
    JUDGE_MEDIAN_RUNS        默认 1；>1 时每条跑 N 次取各维度中位数抑制抖动
    JUDGE_SECOND_MODEL       可选双裁判模型；设置后二次判定并与主裁判比对 pass 一致性
"""
from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from config import config


_JUDGE_SYSTEM_PROMPT = """你是一个严格的 RAG 评测裁判（LLM-as-Judge）。
你的任务：根据给定的参考答案（ground_truth）、检索上下文（contexts）与评分标准（judge_criteria），
对被测系统的回答（answer）打分。评分维度：
1. faithfulness（忠实度 0-5）：回答是否基于检索上下文，有无编造上下文与参考答案之外的事实；出现幻觉必须给低分。
2. relevance（相关性 0-5）：回答是否正面、直接回答用户问题，而不是回避或答非所问。
3. completeness（完整性 0-5）：参考答案中的关键点是否覆盖齐全。
4. citation_quality（引用质量 0-5）：回答是否按要求标注来源（如 [1][2] 或文件名+页码），引用是否与上下文对应。
5. score（总分 0-5）：综合以上维度给出最终分；约 4 分=合格，5 分=优秀。
6. answer_status（答案状态）：若题目提供 expected_answer_status 与 expected_evidence_min：
   - expected=ok：回答必须给出基于上下文的结论，且引用数 >= expected_evidence_min（若有），不足则视为状态不符；
   - expected=insufficient_evidence：资料外/资料缺失题，回答必须如实说明"未提及/未提供/无数据"且不编造；
   - expected=gateway_unavailable：回答应体现网关不可用降级。
   状态不符时 completeness 与 score 应显著扣分。
7. keyword_variants（关键字变体）：若题目提供 keyword_variants（expected_keywords 的跨标准变体写法，
   例如 BCI=大电流注入=Bulk Current Injection、V_BR=击穿电压=Breakdown Voltage、PPK=峰值脉冲功率），
   回答中出现任一等价变体、单位换算（如 8/20us 与 8/20μs）或中英混写即视为命中，
   不得因措辞与参考答案不同而误判为未命中。
规则：
- 必须到 contexts（原文片段）中核对数值与结论：回答给出的数值/结论能在原文中找到对应即视为正确，
  即使写法不同（单位、缩写、中文/英文）也判命中；不得凭常识或记忆补全原文没有的内容。
- 若上下文明确包含答案而回答却说"资料中未提及"，faithfulness 与 completeness 应给低分。
- 若题目本身是资料外信息（如价格/库存），诚实回答"未提及/无数据"不得扣分，编造才扣分。
- 边界/edge_case 题（judge_criteria 说明"应诚实说明未提及"）按该标准评判。
- expected_evidence_min > 0 的题目，若引用（[n]）数量不足该值，citation_quality 与 score 应扣分。
- 重要校准：ground_truth 仅作参考答案，不代表检索上下文一定包含它。只有在上下文确实
  包含某数据时，才因回答未给出该数据扣分；若上下文没有该数据，回答"资料中未提及"
  不算回答错误（那是检索层的问题，不要惩罚回答本身，可在 feedback 中提示检索缺失）。
- 若 contexts 为空或题目 doc_group 为 none（场景/路由/概念类），不要求引用来源：
  citation_quality 按不适用处理（此项不打分），faithfulness 按"是否编造/是否合理"判断，
  completeness 按是否覆盖参考答案中的关键行为/要素判断，重点看回答是否结构化、可执行。
- 必须输出合法 JSON，不要输出任何其他文字。
输出格式：
{"score": 0-5, "faithfulness": 0-5, "relevance": 0-5, "completeness": 0-5,
 "citation_quality": 0-5（不适用时为 null）, "pass": true/false, "feedback": "一句话改进建议（中文）"}"""


class DeepSeekJudge:
    """基于 DeepSeek API 的 RAG 回答评判器。"""

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        )
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.model_version = os.getenv("DEEPSEEK_MODEL_VERSION", "")
        self.timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "60"))
        self.pass_threshold = float(os.getenv("JUDGE_PASS_THRESHOLD", "4.0"))
        self.max_workers = int(os.getenv("JUDGE_MAX_WORKERS", "4"))
        self.median_runs = int(os.getenv("JUDGE_MEDIAN_RUNS", "1"))
        self.second_model = os.getenv("JUDGE_SECOND_MODEL", "").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ----------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------

    def judge_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """并发评判多条。items 每条含 question/answer/contexts/ground_truth/
        expected_keywords/judge_criteria（缺失字段自动降级）。"""
        if not self.available:
            return [
                {
                    "score": None,
                    "pass": False,
                    "error": "缺少 DEEPSEEK_API_KEY，裁判未运行",
                    "feedback": "",
                }
                for _ in items
            ]

        results: list[dict[str, Any]] = [None] * len(items)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="judge") as pool:
            futures = {pool.submit(self._judge_stable, item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[idx] = {
                        "score": None, "pass": False,
                        "error": f"裁判调用异常: {e}", "feedback": "",
                }
        return results

    def _judge_stable(self, item: dict[str, Any]) -> dict[str, Any]:
        """稳定判定：按 JUDGE_MEDIAN_RUNS 多次运行取各维度中位数；可选第二裁判仲裁。"""
        runs = [self._judge_one(item) for _ in range(max(1, self.median_runs))]
        valid = [r for r in runs if r.get("score") is not None]
        if not valid:
            out = dict(runs[0]) if runs else {}
            out.update(
                {
                    "judge_model": self.model,
                    "judge_model_version": self.model_version,
                    "arbitrated": None,
                }
            )
            return out
        dims = ("score", "faithfulness", "relevance", "completeness", "citation_quality")
        med: dict[str, Any] = {}
        for d in dims:
            vals = sorted(r.get(d) for r in valid if r.get(d) is not None)
            med[d] = vals[len(vals) // 2] if vals else None
        out = dict(valid[len(valid) // 2])
        out.update(med)
        out["judge_model"] = self.model
        out["judge_model_version"] = self.model_version
        out["judge_runs"] = len(valid)
        if med.get("score") is not None:
            out["pass"] = bool(med["score"] >= self.pass_threshold)
        if self.second_model:
            try:
                second = self._judge_one(item, model=self.second_model)
                out["judge_second_model"] = self.second_model
                out["arbitrated"] = bool(second.get("pass") == out.get("pass"))
            except Exception:  # noqa: BLE001
                out["arbitrated"] = None
        else:
            out["arbitrated"] = True
        return out

    def _judge_one(self, item: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        payload = {
            "question": item.get("question", ""),
            "ground_truth": item.get("ground_truth", ""),
            "answer": item.get("answer", ""),
            # 裁判必须看到与生成一致的完整上下文，才能核对 [n] 引用与完整性
            "contexts": (item.get("contexts") or [])[: config.JUDGE_CONTEXTS_MAX],
            "expected_keywords": item.get("expected_keywords") or [],
            "judge_criteria": item.get("judge_criteria", ""),
            "doc_group": item.get("doc_group", ""),
            "expected_answer_status": item.get("expected_answer_status"),
            "expected_evidence_min": item.get("expected_evidence_min"),
            "keyword_variants": item.get("keyword_variants") or [],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        body = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        # DeepSeek official API: disable thinking explicitly, otherwise reasoning fills max_tokens and content is empty.
        # DashScope (Bailian) uses enable_thinking instead.
        if "api.deepseek.com" in self.base_url:
            body["thinking"] = {"type": "disabled"}
        else:
            body["enable_thinking"] = False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=body, timeout=self.timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
                content = resp.json()["choices"][0]["message"]["content"]
                result = json.loads(content)
                result.setdefault("score", 0)
                result.setdefault("faithfulness", 0)
                result.setdefault("relevance", 0)
                result.setdefault("completeness", 0)
                result.setdefault("citation_quality", 0)
                result["score"] = float(result.get("score", 0))
                result["pass"] = bool(result.get("pass", result["score"] >= self.pass_threshold))
                result.setdefault("feedback", "")
                if item.get("doc_group") == "none" or not item.get("contexts"):
                    # 场景/路由/无上下文题：引用不适用
                    result["citation_quality"] = None
                    note = "（无知识库上下文，引用不适用）"
                    if note not in result.get("feedback", ""):
                        result["feedback"] = (result.get("feedback", "") + " " + note).strip()
                result.pop("error", None)
                return result
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    # 第二次尝试不强制 JSON 输出（部分模型不支持 response_format）
                    body.pop("response_format", None)
                    continue
                return {
                    "score": None, "pass": False,
                    "error": str(e)[:200], "feedback": "",
                }
        return {"score": None, "pass": False, "error": "unknown", "feedback": ""}


def summarize_judge(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总裁判结果：平均分、通过率、各维度均值与标准差。"""
    judged = [r for r in results if r.get("score") is not None]
    if not judged:
        return {"judged": 0, "note": "裁判未运行（缺少 DEEPSEEK_API_KEY 或全部失败）"}
    dims = ("score", "faithfulness", "relevance", "completeness", "citation_quality")
    avg = {}
    std = {}
    for d in dims:
        vals = [r.get(d) for r in judged if r.get(d) is not None]
        avg[d] = round(sum(vals) / len(vals), 3) if vals else None
        std[d] = round(statistics.stdev(vals), 3) if len(vals) > 1 else None
    models = sorted({r.get("judge_model", "") for r in judged if r.get("judge_model")})
    return {
        "judged": len(judged),
        "pass_count": sum(1 for r in judged if r.get("pass")),
        "pass_rate": round(sum(1 for r in judged if r.get("pass")) / len(judged), 3),
        "avg": avg,
        "std": std,
        "judge_models": models,
    }


def load_env(root_env: Path | None = None) -> None:
    """可选：显式加载根 .env（供单独运行本模块调试用）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(root_env or Path(__file__).resolve().parents[3] / ".env")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    load_env()
    judge = DeepSeekJudge()
    print("available:", judge.available, "| model:", judge.model, "| base:", judge.base_url)
