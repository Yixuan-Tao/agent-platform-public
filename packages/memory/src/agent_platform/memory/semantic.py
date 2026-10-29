"""Opt-in semantic-memory comparison using an explicitly supplied existing embedder."""
import math


class SemanticMemory:
    def __init__(self, embed):
        self.embed = embed

    async def search(self, query, records, *, owner, data_version, k=3):
        eligible = [record for record in records
            if ((record.get("owner") == owner and record.get("scope") == "private" and record.get("state") == "verified")
                or (record.get("scope") == "shared" and record.get("state") == "published"))
            and record.get("data_version") == data_version]
        if not eligible:
            return []
        vectors = await self.embed([query] + [record["question"] for record in eligible])
        if len(vectors) != len(eligible) + 1 or any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("Embedding result shape mismatch")
        def cosine(a, b):
            denom = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
            return sum(x*y for x, y in zip(a, b)) / denom if denom else 0
        return sorted([dict(record, score=cosine(vectors[0], vector)) for record, vector in zip(eligible, vectors[1:])],
                      key=lambda r: r["score"], reverse=True)[:k]
