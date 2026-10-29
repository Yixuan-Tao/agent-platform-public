# Test values are synthetic fixtures for software behavior.
"""单查询冒烟测试 — 验证 Pipeline 基本功能"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.query import QueryPipeline

q = "DEMO27-A 的击穿电压（V_BR）是多少？"
print(f"问题: {q}")
t0 = time.time()
pipeline = QueryPipeline()
print(f"Pipeline 初始化耗时: {time.time()-t0:.1f}s")

t0 = time.time()
result = pipeline.query(q, use_rerank=True, language='zh')
print(f"查询耗时: {time.time()-t0:.1f}s")
print(f"答案: {result.get('answer', '')[:300]}")
print(f"sources 数: {len(result.get('sources', []))}")
