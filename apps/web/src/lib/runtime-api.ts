export const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("agent_platform_token");
  const key = localStorage.getItem("agent_platform_api_key");
  return token ? { Authorization: `Bearer ${token}` } : key ? { "X-API-Key": key } : {};
}

export async function runtimeAPI<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(apiBase + "/api/v1" + path, {
    ...options, headers: { "Content-Type": "application/json", ...authHeaders(), ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败 (${response.status})`);
  }
  return response.json();
}

export async function downloadArtifact(run: string, artifact: string, name: string) {
  const response = await fetch(`${apiBase}/api/v1/runs/${run}/artifacts/${artifact}`, { headers: authHeaders() });
  if (!response.ok) throw new Error("无法读取产物");
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  URL.revokeObjectURL(url);
}
