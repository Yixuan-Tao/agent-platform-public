"""轮询摄入任务状态，写入 poll_log.txt"""
import requests, time, json

task_ids = ['9e82900bb3f8', '5f799adc9bb7', '634b6ca9552a', 'ab08cf7d1795']
pending = set(task_ids)
start = time.time()

# 清空日志
open('poll_log.txt', 'w', encoding='utf-8').close()

while pending and (time.time() - start) < 1200:
    time.sleep(15)
    elapsed = int(time.time() - start)
    lines = []
    for tid in list(pending):
        try:
            r = requests.get(f'http://localhost:8000/ingest/status/{tid}', timeout=10)
            d = r.json()
            s = d.get('status', '')
            p = d.get('progress', '')
            fn = d.get('file_name', '')[:30]
            if s == 'done':
                pending.discard(tid)
                rc = d.get('result', {})
                lines.append(f'[{elapsed}s] [{tid[:8]}] DONE | chunks={rc.get("chunk_count","?")} | {fn}')
            elif s == 'error':
                pending.discard(tid)
                lines.append(f'[{elapsed}s] [{tid[:8]}] ERROR | {d.get("error","")[:80]} | {fn}')
            else:
                lines.append(f'[{elapsed}s] [{tid[:8]}] {s} | {p[:30]} | {fn}')
        except Exception as e:
            lines.append(f'[{elapsed}s] [{tid[:8]}] EXC: {e}')
    with open('poll_log.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

# 最终汇总
with open('poll_log.txt', 'a', encoding='utf-8') as f:
    if pending:
        f.write(f'\n超时未完成: {pending}\n')
    else:
        f.write('\n=== 全部摄入完成 ===\n')
    # 获取文档列表
    r = requests.get('http://localhost:8000/documents', timeout=10)
    docs = r.json()
    f.write(f'已索引文档数: {docs.get("total", 0)}\n')
    for d in docs.get('documents', []):
        f.write(f'  - {d["file_name"][:50]} | chunks={d.get("chunk_count", "?")}\n')
