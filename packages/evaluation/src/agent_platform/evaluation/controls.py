"""Fixed governance controls execute code assertions in a separate trusted test process."""
import asyncio
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CONTROL_TESTS = {
    "用户记忆隔离": "test_private_candidates_and_legacy_memory_never_cross_owner",
    "反馈撤销归属": "test_memory_revoke_persisted_source_not_only_pending",
    "注入不能扩权": "test_budget_prevents_tool_side_effect_and_skill_cannot_grant_scope",
    "工具枚举验证": "test_full_nested_json_schema",
    "嵌套对象验证": "test_full_nested_json_schema",
    "预算预留": "test_budget_recovery_never_resets",
    "重试计费": "test_each_retry_reserves_model_call_and_emits_unknown_usage",
    "取消围栏": "test_owner_session_seriality_cancel_and_fencing",
    "checkpoint恢复": "test_approval_resume_duplicate_click_executes_write_once",
    "租约过期": "test_expired_lease_reclaimed_old_worker_cannot_publish",
    "浏览器断连": "test_complete_worker_checkpoint_artifacts_and_replay",
    "重复写调用": "test_approval_resume_duplicate_click_executes_write_once",
    "流式失败不重放": "test_stream_fails_before_tokens_without_replaying_graph",
    "Checker异常": "test_checker_failure_is_unverified_not_completed",
    "RAG异常": "test_rag_search_failure_returns_friendly",
    "追踪导出异常": "test_tracing_export_failure_does_not_erase_local_event",
    "技能路径穿越": "test_skill_path_guard_versions_and_context_preserves_required",
    "未知写结果暂停": "test_unknown_write_blocks_resume_and_requires_verified_reconciliation",
    "同会话串行": "test_owner_session_seriality_cancel_and_fencing",
    "上下文必要证据": "test_skill_path_guard_versions_and_context_preserves_required",
    "RAG跨服务预算": "test_remote_grant_settlement_is_atomic_fenced_and_idempotent",
    "RAG未知用量恢复": "test_concurrent_grants_and_crash_keep_unknown_consumption_on_recovery",
    "RAG语料失效": "test_worker_with_real_rag_rpc_accounts_usage_and_stops_changed_index",
    "RAG回执校验": "test_unknown_or_invalid_receipt_keeps_entire_reservation",
}

async def run_controls(batch_id):
    directory = ROOT / "runtime_data" / "experiments" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "controls.xml"
    paths = ["packages/orchestrator/tests/test_runtime.py", "packages/orchestrator/tests/test_streaming.py",
             "packages/tools/tests/test_governance.py", "packages/tools/tests/test_tools.py", "packages/memory/tests/test_memory.py", "packages/core/tests/test_telemetry.py"]
    paths += ["packages/tools/tests/test_rag_contract.py", "packages/orchestrator/tests/test_rag_runtime_integration.py"]
    environment = {**os.environ, "LLM_API_KEY": "", "PYTHONIOENCODING": "utf-8"}
    environment.pop("PLATFORM_TEST_DATABASE_URL", None)
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "pytest", *paths, "-q", f"--junitxml={report}",
                    cwd=ROOT, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 180)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    (directory / "controls.log").write_bytes(output)
    if not report.exists():
        return {"passed": False, "error": "控制测试未生成结果"}
    cases = list(ET.parse(report).iter("testcase"))
    statuses = {}
    for control, name in CONTROL_TESTS.items():
        matching = [case for case in cases if case.attrib["name"].split("[")[0] == name]
        statuses[control] = "pass" if matching and all(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in matching) else "fail"
    return {"passed": process.returncode == 0 and all(s == "pass" for s in statuses.values()),
            "results": statuses, "report": str(report), "log": str(directory / "controls.log")}
