# RAG service

Independent FastAPI service for document ingestion, retrieval, reranking, generation, evidence and document comparison. Install `requirements.txt`, configure the repository-root `.env`, and run `python main.py` from this directory.

No document corpus, vector index or account database is bundled. Ingest your own authorized documents after startup. The document-family registry and `resources/component-parameters.csv` are synthetic demonstrations; they do not describe actual certification requirements or replacement decisions.

Run offline tests with `python -m pytest tests/unit -q`. Real API/integration tests require separately configured services and are not evidence of model quality by themselves. See [evaluation/README.md](evaluation/README.md) for evaluation tools and [../../PUBLIC_CONTENT.md](../../PUBLIC_CONTENT.md) for content rules.
