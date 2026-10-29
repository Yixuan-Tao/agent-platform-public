# Public content policy

This repository contains source code, synthetic software fixtures and generic templates. It intentionally starts with a new Git history.

Do not commit credentials, actual `.env` files, identity/contact records, user documents, knowledge corpora, private standard extracts, model responses, internal service addresses, databases, vector indexes, logs or parser caches. Keep operational data outside version control.

Run `python scripts/check_public_content.py` before committing. The check inspects tracked files when Git is initialized and rejects common secret patterns, personal home paths and operational artifacts. Review new fixtures and documents manually as well. Automated checks cannot establish that every piece of prose or business logic is publishable.

Development database credentials, reserved example domains and explicitly synthetic authentication fixtures are examples for software tests. They are not service credentials. Replace development defaults before deployment.
