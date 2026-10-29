# Test values are synthetic fixtures for software behavior.
"""单查询测试 - 不使用 rerank"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.query import QueryPipeline

pipeline = QueryPipeline()
q = "DEMO27-A 的击穿电压"
print(f"问题: {q}")
t0 = time.time()
result = pipeline.query(q, use_rerank=False, language='zh')
print(f"查询耗时: {time.time()-t0:.1f}s")
print(f"答案: {result.get('answer', '')[:400]}")
print(f"sources 数: {len(result.get('sources', []))}")
for i, s in enumerate(result.get('sources', [])[:3]):
    print(f"  [{i+1}] {s.get('file_name','')[:30]} p.{s.get('page','')} | {s.get('content','')[:200]}")
