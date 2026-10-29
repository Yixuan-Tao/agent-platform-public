# Test values are synthetic fixtures for software behavior.
"""Isolated real Chroma probe, unaffected by unit-test module mocks."""
import json
import sys

from db.chroma_client import ChromaClientSingleton
from utils.corpus_snapshot import runtime_snapshot

client = ChromaClientSingleton(sys.argv[1])
client.connect()
system = client._client._system  # This process owns only its temporary test database.
try:
    collection = client._doc_collection
    collection.add(ids=["a"],documents=["30 dBµV/m"],metadatas=[{"version":"2020"}],embeddings=[[1.,0.]])
    first = runtime_snapshot(client)
    assert first["collections"]["documents"]["count"] == 1
    assert first == runtime_snapshot(client)
    collection.update(ids=["a"],documents=["31 dBµV/m"],embeddings=[[0.,1.]])
    second = runtime_snapshot(client)
    assert second["collections"]["documents"]["count"] == 1
    assert second["corpus_hash"] != first["corpus_hash"]
    print(json.dumps({"count":1,"stable_before":True,"changed_after_equal_count_update":True}))
finally:
    system.stop()
    client.close()
