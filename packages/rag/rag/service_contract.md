# RagService 数据契约

> 本文件记录了 `RagService` 所有公共方法的输入/输出结构。
>
> **修改 RagService 内部实现时，必须保持以下返回结构不变**，
> 否则下游工具（`tools/`）可能静默出错。

---

## 1. 通用：标准化 chunk dict

所有方法返回的 chunk 都经过 `Retriever._normalize_hit()` 规整，字段如下：

```python
{
    # ── 标识 ──
    "id": str,                # chunk 唯一 ID（ChromaDB 主键）
    "doc_id": str,            # 文档 ID（md5 文件哈希）
    "file_name": str,         # 文件名（含扩展名）

    # ── 内容 ──
    "content": str,           # chunk 文本内容（table 模态为 Markdown）
    "searchable_text": str,   # 可检索文本（图片为 VL 描述文本）
    "section": str,           # 段落标题（如 "5.3 BCI 测试"）
    "page": int,              # 页码（1-based）

    # ── 模态 ──
    "modality": str,          # "text" | "table" | "image"
    "metadata": dict,         # 原始 metadata（含 child_images, language, file_ext 等）

    # ── 关联 ──
    "child_images": list[str],       # 该 chunk 关联的图片 ID 列表
    "parent_chunk_id": str,          # 父 chunk ID（parent-child 结构）
    "linked_chunks": list[dict],     # 图文关联增强后的关联 chunk 摘要

    # ── 章节路由（对比用） ──
    "section_id": str,        # 层级化编号，如 "5", "5.3", "5.3.2"
    "section_path": str,      # 完整路径，如 "5.3.2 / BCI Test"
    "section_topic": str,     # 章节主题，如 "immunity.bci"

    # ── 父子结构增强（_link_related 后动态添加） ──
    "parent_context": str,    # 父 chunk 内容摘要（前 600 字符）
    "parent_section": str,    # 父 chunk 的 section 标题

    # ── 检索评分 ──
    "score": float,           # 混合检索融合分数（全量拉取时为 1.0）
}
```

### linked_chunks 中的单条条目

```python
{
    "id": str,          # 关联 chunk 的 ID
    "modality": str,    # "text" | "table" | "image"
    "file_name": str,   # 文件名
    "page": int,        # 页码
    "content": str,     # 关联 chunk 的内容摘要
}
```

---

## 2. 方法签名与返回结构

### `retrieve(query, top_k=20, filter_expr=None)`

```python
def retrieve(
    query: str,
    top_k: int = 20,
    filter_expr: Optional[str] = None,
) -> list[dict]:
```

- **用途**: 混合检索（稠密 + 稀疏 + 重排 + 图文关联）
- **返回**: 标准化 chunk dict 列表，按 score 降序
- **路径**: `retriever.retrieve()` → `hybrid_search()` → `_normalize_hit()` → `_link_related()`

### `get_document_chunks(doc_id)`

```python
def get_document_chunks(doc_id: str) -> list[dict]:
```

- **用途**: 按 doc_id 全量拉取所有 chunk
- **返回**: 标准化 chunk dict 列表，按 (page, modality) 升序
- **路径**: `chroma.get_all_by_doc_id()` → `_normalize_hit()`
- **注意**: 无 score 概念，统一设为 `1.0`

### `get_chunk_by_id(chunk_id)`

```python
def get_chunk_by_id(chunk_id: str) -> Optional[dict]:
```

- **用途**: 按 chunk_id 回查单条
- **返回**: 标准化 chunk dict，未找到返回 `None`
- **路径**: `chroma.get_by_id()` → `_build_entity()`

### `list_documents()`

```python
def list_documents() -> list[dict]:
```

- **返回**: 每个元素为 `{"file_name": str, "chunk_count": int}`
- **缓存**: 带 30 秒 TTL

### `search_by_keyword(keyword, doc_id=None)`

```python
def search_by_keyword(
    keyword: str,
    doc_id: Optional[str] = None,
) -> list[dict]:
```

- **用途**: 按关键词搜索（复用 `retrieve()`）
- **返回**: 同 `retrieve()` 的标准化 chunk 列表

---

## 3. 下游消费者一览

| 方法 | EmcCompareTool | ComponentCheckTool | EmcVisualizeTool | API 路由 |
|------|---------------|-------------------|-----------------|---------|
| `retrieve()` | 直接调用 retriever | — | ✅ `query()`（封装） | — |
| `get_document_chunks()` | 直接调用 chroma | ✅ | — | — |
| `retrieve_all_by_doc()` | — | — | ✅ | — |
| `search_by_keyword()` | — | — | — | — |
| `list_documents()` | — | — | — | — |

> **注意**: `EmcCompareTool` 不使用 `RagService`，它直接注入 `retriever` 和 `chroma`。
> 如果要改 `RagService` 导致这两个组件实例化方式变化，需要同步修改 `EmcCompareTool` 的构造。
