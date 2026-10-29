"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { runtimeAPI } from "../../lib/runtime-api";
type Metrics = { n: number; success_rate: number | null; confidence_interval: number[]; latency_p50: number | null; latency_p95: number | null; tokens_actual: number; tokens_estimated: number; model_requests: number; tool_attempts: number; failures: Record<string, number> };
type Batch = { id: string; status: string; error?: string; results: unknown[]; report?: Record<string, Metrics>; config: { split: string; repetitions: number }; dataset_hash: string };
export default function ExperimentsPage() {
  const [batches, setBatches] = useState<Batch[]>([]);
  const [error, setError] = useState("");
  const [paid, setPaid] = useState(false);
  const [split, setSplit] = useState("development");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => runtimeAPI<Batch[]>("/experiments", { signal: controller.signal }).then(setBatches).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    refresh(); const timer = setInterval(refresh, 5000);
    return () => { clearInterval(timer); controller.abort(); };
  }, []);
  async function create() {
    setBusy(true); setError("");
    try {
      await runtimeAPI("/experiments", { method: "POST", body: JSON.stringify({ split, repetitions: 3, data_version: "synthetic-platform-v1", allow_paid_models: paid }) });
      setBatches(await runtimeAPI<Batch[]>("/experiments"));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function resumeBatch(id: string) {
    setBusy(true); setError("");
    try {
      await runtimeAPI(`/experiments/${id}/resume`, { method: "POST", body: JSON.stringify({ allow_paid_models: paid }) });
      setBatches(await runtimeAPI<Batch[]>("/experiments"));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  return <main className="mx-auto max-w-6xl p-6 space-y-6"><nav className="flex gap-5"><Link href="/">对话</Link><Link href="/tasks">任务</Link><Link href="/experiments" aria-current="page">实验</Link></nav><h1 className="text-2xl font-semibold">实验工作台</h1><p>固定配置比较 Skills、上下文和编排策略。业务动作使用合成系统，真实模型调用消耗额度。</p>
    {error && <p role="alert" className="bg-red-50 p-3 text-red-800">{error}</p>}
    <form onSubmit={e => { e.preventDefault(); create(); }} className="rounded-xl bg-white dark:bg-slate-900 p-5 space-y-4"><label htmlFor="split">数据划分 </label><select id="split" value={split} onChange={e => setSplit(e.target.value)} className="text-slate-900 rounded border p-2"><option value="development">开发集</option><option value="validation">验证集</option><option value="holdout">留出集</option></select><label className="block"><input type="checkbox" checked={paid} onChange={e => setPaid(e.target.checked)} /> 手动启动真实模型调用，接受本批次额度消耗</label><button disabled={!paid || busy} className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50">启动七组对照（每题三次）</button></form>
    {batches.map(batch => <section key={batch.id} className="rounded-xl bg-white dark:bg-slate-900 p-5 space-y-3"><h2>{batch.id} · {batch.status}</h2><p>{batch.config.split} · 已记录 {batch.results.length} 次运行</p>{batch.error && <p role="alert">{batch.error}</p>}<details><summary>数据版本</summary><p className="break-all">{batch.dataset_hash}</p></details>{batch.report && <div className="overflow-auto"><table className="w-full text-left"><thead><tr>{["配置", "样本", "成功率", "95%区间", "P50/P95秒", "实际/估算token", "模型/工具调用"].map(title => <th key={title} className="p-2">{title}</th>)}</tr></thead><tbody>{Object.entries(batch.report).map(([name, m]) => <tr key={name}><td className="p-2">{name}</td><td>{m.n}</td><td>{m.success_rate === null ? "—" : `${(m.success_rate * 100).toFixed(1)}%`}</td><td>{m.confidence_interval.map(v => `${(v*100).toFixed(1)}%`).join("–")}</td><td>{m.latency_p50?.toFixed(2)} / {m.latency_p95?.toFixed(2)}</td><td>{m.tokens_actual} / {m.tokens_estimated}</td><td>{m.model_requests} / {m.tool_attempts}</td></tr>)}</tbody></table></div>}<p className="text-sm text-slate-500">费用未确认；配置保持实验状态，固定治理测试和晋级门槛通过后才可调整默认值。</p></section>)}
    {batches.filter(batch => batch.report).map(batch => <details key={`failures-${batch.id}`} className="rounded border p-4"><summary>失败分类 · {batch.id}</summary>{Object.entries(batch.report || {}).map(([name, metrics]) => <div key={name}><h3 className="mt-3 font-medium">{name}</h3>{Object.entries(metrics.failures).map(([reason, count]) => <p key={reason}>{reason}：{count}</p>)}</div>)}</details>)}
    {batches.filter(batch => ["interrupted", "failed"].includes(batch.status)).map(batch => <div key={`resume-${batch.id}`} className="flex items-center gap-4"><span>{batch.id}</span><button disabled={!paid || busy} onClick={() => resumeBatch(batch.id)} className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50">继续该批次</button><span className="text-sm">沿用冻结配置和已有结果；需先勾选模型额度确认。</span></div>)}
    <p className="text-sm text-slate-500">耗时统计使用活跃执行时间；总墙钟时间另存于原始实验记录。</p>
  </main>;
}
