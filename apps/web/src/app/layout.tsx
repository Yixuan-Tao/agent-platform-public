import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "agent-platform",
  description: "统一企业 Agent 平台",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="bg-slate-100 text-slate-900 antialiased dark:bg-slate-950 dark:text-slate-100">
        <div className="px-5 py-2 text-sm flex gap-4 border-b border-slate-200"><a href="/">对话</a><a href="/tasks">任务工作台</a><a href="/experiments">实验工作台</a></div>
        {children}
      </body>
    </html>
  );
}
