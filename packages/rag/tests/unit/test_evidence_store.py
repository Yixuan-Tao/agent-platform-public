# Test values are synthetic fixtures for software behavior.
from utils.evidence_store import EvidenceItem, EvidenceStore


def test_publishing_a_version_archives_the_previous_published_version(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    first = store.create_version("EMC 基线", "emc")
    store.add_items(first["id"], [EvidenceItem(
        subject_id="CISPR25-2021", domain="emc", field_name="频段",
        value_raw="150 kHz–30 MHz", value_normalized={"min_hz": 150000, "max_hz": 30000000},
        unit="Hz", value_kind="range", source_file="cispr.pdf", page=12, section="5.1",
    )])
    store.publish(first["id"])

    second = store.create_version("EMC 修订", "emc")
    store.publish(second["id"])

    assert store.get_version(first["id"])["status"] == "archived"
    assert store.get_version(second["id"])["status"] == "published"


def test_evidence_item_round_trips_structured_conditions_and_citations(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    version = store.create_version("TVS", "component")
    item = EvidenceItem(
        subject_id="TVS-A", domain="component", field_name="钳位电压",
        value_raw="43 V @ IPP=10 A", value_normalized=43, unit="V", value_kind="max",
        test_condition="IPP=10 A", source_file="tvs-a.pdf", page=4, section="Electrical characteristics",
    )
    store.add_items(version["id"], [item])

    stored = store.list_items(version["id"])
    assert stored[0]["test_condition"] == "IPP=10 A"
    assert stored[0]["citation"] == {"file_name": "tvs-a.pdf", "page": 4, "section": "Electrical characteristics"}


def test_component_rules_are_versioned_with_evidence(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    version = store.create_version("MOSFET", "component")
    store.add_rules(version["id"], [{"parameter": "VDS", "rule": "candidate_gte", "critical": True}])
    assert store.list_rules(version["id"]) == [{"parameter": "VDS", "rule": "candidate_gte", "critical": True}]


def test_archived_version_can_be_reactivated_as_the_published_version(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    first, second = store.create_version("first", "emc"), store.create_version("second", "emc")
    store.publish(first["id"])
    store.publish(second["id"])
    store.activate(first["id"])
    assert store.get_version(first["id"])["status"] == "published"
    assert store.get_version(second["id"])["status"] == "archived"
