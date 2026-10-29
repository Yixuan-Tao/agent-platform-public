"""Synthetic document families for exercising generic alias and family retrieval.

These labels describe invented documents, with no technical requirements or real corpus.
Extend the registry only with data you are authorized to publish.
"""
from __future__ import annotations
import re

def norm(s: str) -> str:
    """归一化：小写 + 去掉空白/横线/下划线/斜杠/括号/点号等分隔符。"""
    return re.sub(r"[\s\-_/\\（）()\[\]【】.．·:：;；,，\"'`]+", "", (s or "").lower())


STANDARD_ALIASES = [("demo alpha", "demoalpha"), ("demo beta", "demobeta")]
STANDARD_FAMILIES = {
    "demo_alpha": {"label": "Synthetic Alpha", "match_groups": [["demoalpha"]], "file_patterns": [["demoalpha"]]},
    "demo_beta": {"label": "Synthetic Beta", "match_groups": [["demobeta"]], "file_patterns": [["demobeta"]]},
}

def alias_file_fragments(query: str) -> list[str]:
    """返回 query 中命中别名对应的文件名片段（归一化后）。"""
    q = norm(query)
    if not q:
        return []
    frags: list[str] = []
    for alias, frag in STANDARD_ALIASES:
        if norm(alias) in q and frag not in frags:
            frags.append(frag)
    return frags


def detect_families(query: str) -> list[str]:
    """检测 query 引用了哪些标准族（任一 match_group 全部命中）。"""
    q = norm(query)
    if not q:
        return []
    hits: list[str] = []
    for key, meta in STANDARD_FAMILIES.items():
        for group in meta["match_groups"]:
            if all(term in q for term in group):
                hits.append(key)
                break
    return hits


def family_member_files(family_key: str, file_names: list[str]) -> list[str]:
    """按文件模式筛选某标准族的成员文档（保持库内顺序）。"""
    meta = STANDARD_FAMILIES.get(family_key)
    if not meta:
        return []
    out: list[str] = []
    for fn in file_names:
        f = norm(fn)
        if any(all(pat in f for pat in group) for group in meta["file_patterns"]):
            out.append(fn)
    return out
