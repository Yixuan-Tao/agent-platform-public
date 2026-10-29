"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_STORAGE = "agent_platform_token";
const USER_STORAGE = "agent_platform_user";
const KEY_STORAGE = "agent_platform_api_key";

interface Msg {
  role: "user" | "assistant";
  content: string;
  agent?: string;
  message_id?: string;
}

interface AgentInfo {
  name: string;
  description: string;
}

interface SessionItem {
  session_id: string;
  title: string;
  updated_at: string;
}

interface HistoryMessage {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  agent?: string;
}

export default function Chat() {
  const [token, setToken] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(TOKEN_STORAGE) || "";
  });
  const [username, setUsername] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(USER_STORAGE) || "";
  });
  const [apiKey, setApiKey] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(KEY_STORAGE) || "";
  });
  const [loginName, setLoginName] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [agentHint, setAgentHint] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dark, setDark] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const keyRef = useRef(apiKey);
  const tokenRef = useRef(token);
  keyRef.current = apiKey;
  tokenRef.current = token;

  /** 统一凭据头：优先使用登录会话令牌，API Key 作为自动化兜底。 */
  const authHeaders = useCallback((): Record<string, string> => {
    if (tokenRef.current) return { Authorization: `Bearer ${tokenRef.current}` };
    if (keyRef.current) return { "X-API-Key": keyRef.current };
    return {};
  }, []);

  const fetchAgents = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/agents`, {
        headers: authHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setAgents(data.agents || []);
      }
    } catch {
      /* 网关未启动时静默 */
    }
  }, [authHeaders]);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/sessions`, {
        headers: authHeaders(),
      });
      if (res.ok) setSessions(await res.json());
    } catch {
      /* ignore */
    }
  }, [authHeaders]);

  const loadHistory = useCallback(
    async (target: string) => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/sessions/${target}/messages`, {
          headers: authHeaders(),
        });
        if (!res.ok) return;
        const data: HistoryMessage[] = await res.json();
        setMessages(
          data.map((m) => ({
            role: m.role,
            content: m.content,
            agent: m.agent || undefined,
            message_id: m.message_id,
          })),
        );
      } catch {
        /* ignore */
      }
    },
    [authHeaders],
  );

  useEffect(() => {
    fetchAgents();
    fetchSessions();
  }, [fetchAgents, fetchSessions]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function saveKey(value: string) {
    setApiKey(value);
    localStorage.setItem(KEY_STORAGE, value);
  }

  function saveToken(value: string, name: string) {
    setToken(value);
    setUsername(name);
    setApiKey("");
    localStorage.setItem(TOKEN_STORAGE, value);
    localStorage.setItem(USER_STORAGE, name);
    localStorage.removeItem(KEY_STORAGE);
  }

  async function login() {
    setLoginError("");
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginName, password: loginPassword }),
      });
      if (!res.ok) {
        setLoginError(res.status === 401 ? "用户名或密码错误" : `登录失败（HTTP ${res.status}）`);
        return;
      }
      const data = await res.json();
      saveToken(data.access_token, loginName);
      setLoginPassword("");
      await Promise.all([fetchAgents(), fetchSessions()]);
    } catch (e: any) {
      setLoginError(`登录请求失败：${e.message}`);
    }
  }

  async function logout() {
    try {
      await fetch(`${API_BASE}/api/v1/auth/logout`, {
        method: "POST",
        headers: authHeaders(),
      });
    } catch {
      /* ignore */
    }
    setToken("");
    setUsername("");
    localStorage.removeItem(TOKEN_STORAGE);
    localStorage.removeItem(USER_STORAGE);
    setSessions([]);
    setMessages([]);
    setSessionId(null);
  }

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setStreaming(true);
    let acc = "";
    let agent = "";
    let answerMessageId = "";
    try {
      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(),
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          agent_hint: agentHint || null,
        }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // 服务端 EventSourceResponse 用 \r\n\r\n 分隔事件；兼容 \n\n
        const parts = buffer.split(/\r?\n\r?\n/);
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          let evt: any;
          try {
            evt = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }
          if (evt.type === "token") acc += evt.content || "";
          else if (evt.type === "agent_switch") agent = evt.agent || agent;
          else if (evt.type === "session") {
            setSessionId(evt.session_id);
            if (evt.message_id) answerMessageId = evt.message_id;
          }
          else if (evt.type === "error") acc += `\n[错误] ${evt.message || ""}`;
          setMessages((m) => {
            const next = [...m];
            if (next.length > 0 && next[next.length - 1].role === "assistant") {
              next[next.length - 1] = {
                role: "assistant",
                content: acc,
                agent,
                message_id: answerMessageId || next[next.length - 1].message_id,
              };
            } else {
              next.push({ role: "assistant", content: acc, agent, message_id: answerMessageId });
            }
            return next;
          });
        }
      }
      fetchSessions();
    } catch (e: any) {
      setMessages((m) => [...m, { role: "assistant", content: `请求失败: ${e.message}` }]);
    } finally {
      setStreaming(false);
    }
  }

  async function sendFeedback(type: "up" | "down", idx: number) {
    const target = messages[idx];
    if (!target?.message_id) {
      alert("该消息缺少可溯源 ID（历史消息或旧会话），暂不能反馈。");
      return;
    }
    const correction = type === "down" ? prompt("请描述问题或修正内容（可选）") : null;
    try {
      const res = await fetch(`${API_BASE}/api/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          session_id: sessionId,
          message_id: target.message_id,
          feedback_type: type,
          correction: correction || undefined,
        }),
      });
      if (!res.ok) alert(`反馈提交失败（HTTP ${res.status}）`);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="flex h-screen">
      {/* 侧栏 */}
      <aside className="hidden w-64 flex-col border-r border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900 md:flex">
        <div className="mb-3 text-sm font-semibold text-slate-500 dark:text-slate-400">Agent 列表</div>
        <div className="space-y-1">
          {agents.map((a) => (
            <button
              key={a.name}
              onClick={() => setAgentHint(agentHint === a.name ? "" : a.name)}
              className={`w-full rounded-lg px-3 py-2 text-left text-sm ${
                agentHint === a.name
                  ? "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200"
                  : "hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
            >
              <div className="font-medium">{a.name}</div>
              <div className="truncate text-xs opacity-70">{a.description}</div>
            </button>
          ))}
        </div>
        <div className="mb-3 mt-6 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-500 dark:text-slate-400">会话</span>
          <button
            onClick={() => {
              setSessionId(null);
              setMessages([]);
            }}
            className="rounded-lg border border-slate-300 px-2 py-0.5 text-xs dark:border-slate-700"
          >
            新会话
          </button>
        </div>
        <div className="flex-1 space-y-1 overflow-auto">
          {sessions.map((s) => (
            <button
              key={s.session_id}
              onClick={() => {
                setSessionId(s.session_id);
                void loadHistory(s.session_id);
              }}
              className={`w-full truncate rounded-lg px-3 py-1.5 text-left text-xs ${
                sessionId === s.session_id
                  ? "bg-slate-200 dark:bg-slate-700"
                  : "hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
            >
              {s.title}
            </button>
          ))}
        </div>
      </aside>

      {/* 主区 */}
      <main className="flex flex-1 flex-col">
        <header className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
          <h1 className="text-lg font-bold">agent-platform</h1>
          <span className="text-xs text-slate-400">{sessionId ? `会话 ${sessionId.slice(0, 8)}` : "新会话"}</span>
          <div className="ml-auto flex items-center gap-2">
            {token ? (
              <>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {username || "已登录"}
                </span>
                <button
                  onClick={logout}
                  className="rounded-lg border border-slate-300 px-2 py-1 text-xs dark:border-slate-700"
                >
                  退出
                </button>
              </>
            ) : (
              <>
                <input
                  value={loginName}
                  onChange={(e) => setLoginName(e.target.value)}
                  placeholder="用户名"
                  className="w-24 rounded-lg border border-slate-300 bg-slate-50 px-2 py-1 text-xs dark:border-slate-700 dark:bg-slate-800"
                />
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") login();
                  }}
                  placeholder="密码"
                  className="w-24 rounded-lg border border-slate-300 bg-slate-50 px-2 py-1 text-xs dark:border-slate-700 dark:bg-slate-800"
                />
                <button
                  onClick={login}
                  disabled={!loginName || !loginPassword}
                  className="rounded-lg bg-sky-600 px-3 py-1 text-xs font-medium text-white disabled:opacity-40"
                >
                  登录
                </button>
                <button
                  onClick={() => setShowAdvanced((v) => !v)}
                  className="rounded-lg border border-slate-300 px-2 py-1 text-xs dark:border-slate-700"
                  title="使用服务 API Key（自动化场景）"
                >
                  服务密钥
                </button>
              </>
            )}
            <button
              onClick={() => setDark(!dark)}
              className="rounded-lg border border-slate-300 px-2 py-1 text-xs dark:border-slate-700"
            >
              {dark ? "浅色" : "深色"}
            </button>
          </div>
          {loginError && (
            <div className="w-full text-xs text-rose-500">{loginError}</div>
          )}
          {showAdvanced && !token && (
            <div className="w-full">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => saveKey(e.target.value)}
                placeholder="X-API-Key（服务密钥，权限受限）"
                className="w-full rounded-lg border border-slate-300 bg-slate-50 px-2 py-1 text-xs dark:border-slate-700 dark:bg-slate-800"
              />
            </div>
          )}
        </header>

        <div className="flex-1 overflow-auto p-4">
          {messages.length === 0 && (
            <div className="mt-20 text-center text-sm text-slate-400">
              输入问题开始对话，例如：VPN 无法连接怎么报修？
            </div>
          )}
          <div className="mx-auto max-w-3xl space-y-4">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
                    m.role === "user"
                      ? "bg-sky-600 text-white"
                      : "bg-white dark:bg-slate-800"
                  }`}
                >
                  {m.role === "assistant" && m.agent && (
                    <div className="mb-1 text-xs font-semibold text-sky-500">{m.agent}</div>
                  )}
                  {m.content}
                  {m.role === "assistant" && !streaming && (
                    <div className="mt-2 flex gap-2 text-xs">
                      <button onClick={() => sendFeedback("up", i)} className="opacity-60 hover:opacity-100">👍</button>
                      <button onClick={() => sendFeedback("down", i)} className="opacity-60 hover:opacity-100">👎</button>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <footer className="border-t border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900">
          <div className="mx-auto flex max-w-3xl items-center gap-2">
            <select
              value={agentHint}
              onChange={(e) => setAgentHint(e.target.value)}
              className="rounded-lg border border-slate-300 bg-slate-50 px-2 py-2 text-xs dark:border-slate-700 dark:bg-slate-800"
            >
              <option value="">自动路由</option>
              {agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行"
              rows={1}
              className="flex-1 resize-none rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-800"
            />
            <button
              onClick={send}
              disabled={streaming || !input.trim()}
              className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              {streaming ? "生成中…" : "发送"}
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}
