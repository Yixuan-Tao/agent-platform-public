"""上传 4 份 PDF 并等待摄入完成"""
import requests, time, os, sys

PDF_DIR = 'example'
BASE_URL = 'http://localhost:8000'
LOG_FILE = 'upload_log.txt'

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# 清空日志
open(LOG_FILE, 'w').close()

# 1) 获取 PDF 列表
pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
log(f'找到 {len(pdfs)} 份 PDF')

# 2) 上传
task_ids = {}
for pdf_name in pdfs:
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    log(f'上传: {pdf_name[:50]}...')
    with open(pdf_path, 'rb') as f:
        resp = requests.post(f'{BASE_URL}/ingest', files={'file': (pdf_name, f)}, timeout=30)
    data = resp.json()
    tid = data['task_id']
    task_ids[tid] = pdf_name
    log(f'  task_id={tid}')

# 3) 轮询等待
pending = set(task_ids.keys())
max_wait = 900
start = time.time()
results = {}

while pending and (time.time() - start) < max_wait:
    time.sleep(10)
    elapsed = int(time.time() - start)
    for tid in list(pending):
        try:
            resp = requests.get(f'{BASE_URL}/ingest/status/{tid}', timeout=10)
            data = resp.json()
            status = data.get('status', '')
            progress = data.get('progress', '')
            fname = task_ids[tid][:40]
            if status in ('done', 'error'):
                pending.discard(tid)
                results[tid] = data
                if status == 'completed':
                    r = data.get('result', {})
                    log(f'[{elapsed}s] [{tid[:8]}] DONE | chunks={r.get("chunk_count","?")} | {fname}')
                else:
                    log(f'[{elapsed}s] [{tid[:8]}] ERROR | {data.get("error","")[:80]} | {fname}')
            else:
                log(f'[{elapsed}s] [{tid[:8]}] {progress[:30]} | {fname}')
        except Exception as e:
            log(f'[{elapsed}s] [{tid[:8]}] 轮询异常: {e}')

if pending:
    log(f'\n超时未完成: {[task_ids[t] for t in pending]}')
else:
    log('\n=== 全部摄入完成 ===')

# 4) 最终确认
resp = requests.get(f'{BASE_URL}/documents', timeout=10)
docs = resp.json()
log(f'已索引文档数: {docs.get("total", 0)}')
for d in docs.get('documents', []):
    log(f'  - {d["file_name"][:50]} | chunks={d.get("chunk_count", "?")}')

# 5) 输出结果摘要
log('\n=== 摄入结果摘要 ===')
for tid, data in results.items():
    fname = task_ids[tid][:50]
    if data.get('status') == 'completed':
        r = data.get('result', {})
        log(f'OK   | {fname} | chunks={r.get("chunk_count","?")}')
    else:
        log(f'FAIL | {fname} | {data.get("error","")[:80]}')
