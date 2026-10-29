"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { apiBase, authHeaders, downloadArtifact, runtimeAPI } from "../lib/runtime-api";

type Approval = { id: string; status: string; tool: string; parameters: Record<string, unknown>; parameters_hash: string; expires_at: number };
type Run = { id: string; session_id: string; goal: string; status: string; budget: Record<string, number>; result: { answer?: string; error?: string; pending?: { question?: string; message?: string; kind?: string; tool_call_id?: string }; verification?: { status: string; reason: string; evidence_refs: unknown[] } }; steps?: { id: string; name: string; status: string }[]; approvals?: Approval[]; artifacts?: { id: string; name: string; size: number }[] };
const statusNames: Record<string, string> = { queued: "排队中", running: "执行中", waiting_input: "等待补充", waiting_approval: "等待审批", completed: "已完成", failed: "未完成", cancelled: "已取消" };
const stepNames: Record<string, string> = { prepare: "准备上下文", execute: "执行任务", verify: "核验证据", produce: "生成产物" };

export default function RunWorkbench() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [profiles, setProfiles] = useState<{ name: string }[]>([]);
  const [goal, setGoal] = useState("");
  const [profile, setProfile] = useState("B0");
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string[]>([]);
  const cursor = useRef(0);
  const selectedId = selected?.id;

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const items = await runtimeAPI<Run[]>("/runs", { signal });
    setRuns(items);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([refresh(controller.signal), runtimeAPI<{ name: string }[]>("/experiment-profiles", { signal: controller.signal }).then(setProfiles)])
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    const timer = setInterval(() => refresh(controller.signal).catch(() => {}), 4000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    cursor.current = 0;
    setProgress([]);
    async function stream() {
      while (!controller.signal.aborted) {
        try {
          const response = await fetch(`${apiBase}/api/v1/runs/${selectedId}/events?after=${cursor.current}`, { headers: authHeaders(), signal: controller.signal });
          if (!response.ok || !response.body) throw new Error("任务进度暂时无法连接");
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (!controller.signal.aborted) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
            const blocks = buffer.split("\n\n"); buffer = blocks.pop() || "";
            for (const block of blocks) {
              const line = block.split("\n").find(l => l.startsWith("data: "));
              if (!line) continue;
              const event = JSON.parse(line.slice(6));
              if (event.seq <= cursor.current) continue;
              cursor.current = event.seq;
              if (["status", "tool", "verification", "skill_selection"].includes(event.type)) {
                const text = event.type === "status" ? statusNames[event.status] || event.status : event.type === "tool" ? `${event.tool}：${event.status || "开始"}` : event.type === "verification" ? `核验：${event.status} ${event.reason || ""}` : `技能：${event.selected.join("、") || "基础流程"}`;
                setProgress(previous => [...previous.slice(-49), text]);
              }
              if (["status", "approval", "artifact", "answer"].includes(event.type)) {
                const current = await runtimeAPI<Run>(`/runs/${selectedId}`, { signal: controller.signal });
                setSelected(current);
              }
            }
          }
          const current = await runtimeAPI<Run>(`/runs/${selectedId}`, { signal: controller.signal });
          setSelected(current);
          if (["completed", "failed", "cancelled"].includes(current.status)) return;
        } catch (e) { if (controller.signal.aborted) return; setError((e as Error).message); }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    stream();
    return () => controller.abort();
  }, [selectedId]);

  async function action(path: string, body: unknown = {}) {
    setBusy(true); setError("");
    try {
      const run = await runtimeAPI<Run>(path, { method: "POST", body: JSON.stringify(body) });
      setSelected(run); await refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  return <main className="mx-auto max-w-6xl p-6 space-y-6">
    <nav className="flex gap-5"><Link href="/">对话</Link><Link href="/tasks" aria-current="page">任务</Link><Link href="/experiments">实验</Link></nav>
    <h1 className="text-2xl font-semibold">任务工作台</h1>
    <p className="text-slate-600">查看执行进度、补充信息和处理审批。关闭页面后任务继续执行。</p>
    {error && <p role="alert" className="rounded bg-red-50 p-3 text-red-800">{error}</p>}
    <form className="rounded-xl bg-white dark:bg-slate-900 p-5 space-y-3" onSubmit={e => { e.preventDefault(); action("/runs", { goal, profile }); }}>
      <label className="block" htmlFor="goal">新任务目标</label>
      <textarea id="goal" required value={goal} onChange={e => setGoal(e.target.value)} className="w-full rounded border p-3 text-slate-900" rows={3} placeholder="例如：对比两个 EMC 标准版本，生成带来源的报告" />
      <label htmlFor="profile">实验配置 </label><select id="profile" value={profile} onChange={e => setProfile(e.target.value)} className="rounded border p-2 text-slate-900">{(profiles.length ? profiles : [{ name: "B0" }]).map(p => <option key={p.name}>{p.name}</option>)}</select>
      <button disabled={busy} className="ml-4 rounded bg-blue-600 px-5 py-2 text-white disabled:opacity-50">创建任务</button>
    </form>
    <div className="grid gap-6 md:grid-cols-[280px_1fr]">
      <section aria-label="任务列表" className="space-y-2">{runs.map(run => <button key={run.id} onClick={() => setSelected(run)} aria-pressed={selectedId === run.id} className="block w-full rounded border p-3 text-left bg-white dark:bg-slate-900"><span className="block truncate">{run.goal}</span><span className="text-sm text-slate-500">{statusNames[run.status] || run.status}</span></button>)}{runs.length === 0 && <p>暂无任务</p>}</section>
      <section className="rounded-xl bg-white dark:bg-slate-900 p-5 space-y-4" aria-label="任务详情">
        {!selected ? <p>选择一个任务查看详情。</p> : <>
          <h2 className="text-xl">{selected.goal}</h2><p>{statusNames[selected.status]} · {selected.id}</p>
          {(selected.steps || []).map(step => <p key={step.id}>{stepNames[step.name] || step.name}：{statusNames[step.status] || step.status}</p>)}
          <details><summary>执行进度</summary><ul>{progress.map((text, i) => <li key={`${i}-${text}`}>{text}</li>)}</ul></details>
          <p>模型请求 {selected.budget.model_requests || 0} · 工具尝试 {selected.budget.tool_attempts || 0} · token预算占用 {selected.budget.tokens || 0}</p>
          {(selected.approvals || []).filter(a => a.status === "pending").map(a => <section key={a.id} className="rounded border border-amber-300 bg-amber-50 p-4 text-slate-900"><h3>需要批准：{a.tool}</h3><pre className="overflow-auto whitespace-pre-wrap">{JSON.stringify(a.parameters, null, 2)}</pre><p>有效期至 {new Date(a.expires_at * 1000).toLocaleString()}</p><button disabled={busy} onClick={() => action(`/runs/${selected.id}/approvals/${a.id}`, { approved: true, parameters_hash: a.parameters_hash })} className="mr-3 rounded bg-blue-600 px-4 py-2 text-white">批准此动作</button><button disabled={busy} onClick={() => action(`/runs/${selected.id}/approvals/${a.id}`, { approved: false, parameters_hash: a.parameters_hash })}>拒绝</button></section>)}
          {selected.result.pending && <p>{selected.result.pending.question || selected.result.pending.message}</p>}
          {selected.result.pending?.kind === "reconcile" && selected.result.pending.tool_call_id && <button disabled={busy} onClick={() => action(`/runs/${selected.id}/tool-calls/${selected.result.pending?.tool_call_id}/reconcile`)} className="rounded border px-4 py-2">查询外部结果并核验</button>}
          {selected.result.error && <p role="alert">{selected.result.error}</p>}
          {selected.result.verification && <p>核验：{selected.result.verification.status} {selected.result.verification.reason}</p>}
          {selected.result.answer && <pre className="whitespace-pre-wrap">{selected.result.answer}</pre>}
          <div className="flex flex-wrap gap-3">{(selected.artifacts || []).map(a => <button key={a.id} onClick={() => downloadArtifact(selected.id, a.id, a.name).catch(e => setError(e.message))} className="rounded border px-3 py-2">下载 {a.name}</button>)}</div>
          {["waiting_input", "failed"].includes(selected.status) && <form onSubmit={e => { e.preventDefault(); action(`/runs/${selected.id}/resume`, { inputs: { answer: input } }); }}><label htmlFor="resume">补充信息或恢复说明</label><textarea id="resume" value={input} onChange={e => setInput(e.target.value)} className="w-full border rounded p-2 text-slate-900" /><button disabled={busy} className="rounded bg-blue-600 px-4 py-2 text-white">恢复任务</button></form>}
          {!["completed", "failed", "cancelled"].includes(selected.status) && <button disabled={busy} onClick={() => action(`/runs/${selected.id}/cancel`)} className="rounded border px-4 py-2">取消任务</button>}
        </>}
      </section>
    </div>
  </main>;
}
