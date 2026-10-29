# Test values are synthetic fixtures for software behavior.
from agent_platform.memory.semantic import SemanticMemory

async def test_semantic_filters_owner_verification_and_version_before_embedding():
    calls = []
    async def embed(texts):
        calls.append(texts)
        return [[1, 0] for text in texts]
    base = {"owner": "alice", "scope": "private", "state": "verified", "data_version": "v1", "question": "VPN", "answer": "restart"}
    records = [base, {**base, "owner": "bob", "question": "bob private"}, {**base, "state": "candidate", "question": "unverified"}, {**base, "data_version": "v0", "question": "old data"}]
    results = await SemanticMemory(embed).search("VPN issue", records, owner="alice", data_version="v1")
    assert calls == [["VPN issue", "VPN"]]
    assert len(results) == 1 and results[0]["owner"] == "alice"
