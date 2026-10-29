# Test values are synthetic fixtures for software behavior.
from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from api.evidence_routes import router
from utils.access_control import AccessStore
from utils.evidence_store import EvidenceStore


def _admin_client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.access_store = AccessStore(tmp_path / "access.db")
    app.state.access_store.create_user("admin", "safe password", "admin")
    app.state.evidence_store = EvidenceStore(tmp_path / "evidence.db")
    token = app.state.access_store.authenticate("admin", "safe password")
    return TestClient(app), {"Authorization": f"Bearer {token}"}


def _workbook_bytes():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "EMC证据"
    sheet.append(["subject_id", "field_name", "value_raw", "value_kind", "source_file", "page", "section", "test_item"])
    sheet.append(["CISPR25", "频段", "150 kHz–30 MHz", "range", "cispr.pdf", 12, "5.1", "CE"])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_admin_can_import_and_publish_evidence_workbook(tmp_path):
    client, headers = _admin_client(tmp_path)
    response = client.post(
        "/evidence/import?name=EMC 基线", headers=headers,
        files={"file": ("emc.xlsx", _workbook_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    version_id = response.json()["id"]
    assert client.post(f"/evidence/versions/{version_id}/publish", headers=headers).status_code == 200
    versions = client.get("/evidence/versions", headers=headers).json()["versions"]
    assert versions[0]["status"] == "published"


def test_viewer_cannot_import_evidence_workbook(tmp_path):
    client, _ = _admin_client(tmp_path)
    app = client.app
    app.state.access_store.create_user("viewer", "viewer password", "viewer")
    token = app.state.access_store.authenticate("viewer", "viewer password")
    response = client.post("/evidence/import?name=x", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
