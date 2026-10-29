# Test values are synthetic fixtures for software behavior.
"""Real lifespan connects the ledger, saver and project skill directory, without model calls."""
from fastapi.testclient import TestClient
from agent_platform.core.config import Settings
from app.main import create_app


def test_enabled_runtime_startup_loads_real_skills_without_paid_calls(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}",
                        runtime_dir=str(tmp_path / "runtime"), chroma_dir=str(tmp_path / "chroma"),
                        log_dir=str(tmp_path / "logs"), runs_enabled=True, api_key="dev-key-change-me")
    class Router:
        def resolve_model(self, agent=None):
            return "fixture-model"
        async def chat(self, *args, **kwargs):
            raise AssertionError("Startup must not call a model")
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.get_llm_router", lambda: Router())
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/experiment-profiles", headers={"X-API-Key": "dev-key-change-me"})
        assert response.status_code == 200
        profiles = response.json()
        assert len(profiles) == 7
        assert {skill["name"] for skill in profiles[0]["skills"]} == {"emc-standard-compare", "it-incident-ticket"}
