"""
对比和可视化 API 路由 (Task 7)

端点：
    - POST /compare/emc         EMC 标准对比
    - POST /compare/component   元器件替换判断
    - POST /compare/visualize   EMC 表格可视化

依赖获取方式（参考 main.py / routes.py）：
    - query_pipeline 通过 request.app.state.query_pipeline 获取
    - retriever / chroma / generator 均为 query_pipeline 的懒加载属性
    - config 为全局配置对象
"""

from fastapi import APIRouter, HTTPException, Request
from uuid import uuid4

from api.schemas import (
    EmcCompareRequest, EmcCompareResponse,
    ComponentCompareRequest, ComponentCompareResponse,
    VisualizeEmcRequest, VisualizeEmcResponse,
)
from config import config
from tools.component_check import ComponentCheckTool
from tools.emc_compare import EmcCompareTool
from tools.emc_visualize import EmcVisualizeTool
from tools.evidence_compare import compare_emc_evidence
from tools.component_evidence_rules import compare_component_evidence
from utils.logger import log

router = APIRouter(prefix="/compare", tags=["compare"])


# ============================================================
# POST /compare/emc — EMC 标准对比
# ============================================================

@router.post("/emc", response_model=EmcCompareResponse)
async def compare_emc(
    request: EmcCompareRequest,
    req: Request,
) -> EmcCompareResponse:
    """EMC 标准对比

    从两份文档中分别提取参数表，基于参数表生成结构化对比报告。
    支持章节路由（section routing）精准定位和全量文档两阶段对比两种模式。
    """
    try:
        # ---- 证据优先：已发布的人工证据版本存在时走确定性对比 ----
        store = getattr(req.app.state, "evidence_store", None)
        if store is not None:
            version = store.get_version(request.evidence_version) if request.evidence_version else store.published_version("emc")
            if version is not None:
                if version["status"] != "published":
                    raise HTTPException(status_code=422, detail="运行时只能使用已发布的证据版本")
                evidence_result = compare_emc_evidence(
                    store.list_items(version["id"]), store.list_items(version["id"]),
                    request.doc_a, request.doc_b,
                    request.test_item if request.comparison_mode == "test_item" else "",
                )
                return EmcCompareResponse(
                    answer=evidence_result["answer"], doc_a={"name": request.doc_a, "params": {}}, doc_b={"name": request.doc_b, "params": {}},
                    comparison_items=evidence_result["comparison_matrix"], comparison_matrix=evidence_result["comparison_matrix"],
                    citations=evidence_result["citations"], answer_status=evidence_result["answer_status"],
                    coverage=evidence_result["coverage"], evidence_version=version["id"], run_id=f"emc-{uuid4().hex}", compare_route="evidence",
                )
        query = request.query
        if request.comparison_mode == "test_item":
            query = f"{query}；仅对比 EMC 测试项：{request.test_item}"
            if request.section_hint:
                query += f"；章节提示：{request.section_hint}"
        query_pipeline = req.app.state.query_pipeline
        tool = EmcCompareTool(
            rag_service=query_pipeline.rag_service,
            generator=query_pipeline.generator,
            config=config,
        )
        result = tool.compare(
            query=query,
            doc_a=request.doc_a,
            doc_b=request.doc_b,
            language=request.language,
        )

        # 构建响应
        doc_a_info = {
            "name": result.get("doc_a", request.doc_a),
            "params": result.get("params_a", {}),
        }
        doc_b_info = {
            "name": result.get("doc_b", request.doc_b),
            "params": result.get("params_b", {}),
        }

        sources = result.get("sources", [])
        citations = [
            {key: source.get(key) for key in ("doc_id", "file_name", "page", "chunk_id", "section", "content") if source.get(key) is not None}
            for source in sources
        ]
        status = "ok" if sources else "insufficient_evidence"
        return EmcCompareResponse(
            answer=result.get("answer", ""),
            doc_a=doc_a_info,
            doc_b=doc_b_info,
            compare_route=result.get("compare_route", ""),
            sources=sources,
            image_chunks=result.get("image_chunks", []),
            comparison_items=result.get("sections", []),
            citations=citations,
            answer_status=status,
            run_id=f"emc-{uuid4().hex}",
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("/compare/emc 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# POST /compare/component — 元器件替换判断
# ============================================================

@router.post("/component", response_model=ComponentCompareResponse)
async def compare_component(
    request: ComponentCompareRequest,
    req: Request,
) -> ComponentCompareResponse:
    """元器件替换判断

    接收用户传入的参数表模板，对两份 datasheet 文档做逐字段严格对比。
    输出：可替换 / 有条件替换 / 不可替换 三档结论。
    """
    try:
        # ---- 证据优先：已发布的人工证据与替代规则存在时走确定性双向判定 ----
        store = getattr(req.app.state, "evidence_store", None)
        if store is not None:
            version = store.get_version(request.evidence_version) if request.evidence_version else store.published_version("component")
            if version is not None:
                if version["status"] != "published":
                    raise HTTPException(status_code=422, detail="运行时只能使用已发布的证据版本")
                result = compare_component_evidence(
                    store.list_items(version["id"]), store.list_items(version["id"]),
                    request.doc_a, request.doc_b, store.list_rules(version["id"]),
                )
                a_to_b, b_to_a = result["A_to_B"], result["B_to_A"]
                citations = a_to_b["citations"] + b_to_a["citations"]
                statuses = {a_to_b["conclusion"], b_to_a["conclusion"]}
                return ComponentCompareResponse(
                    conclusion=a_to_b["conclusion"], conclusion_detail="基于已发布人工证据与替代规则的双向判定",
                    directional_conclusions=result, citations=citations,
                    answer_status="ok" if statuses == {"可替换"} else "insufficient_evidence" if "有条件替换" in statuses else "ok",
                    run_id=f"component-{uuid4().hex}", evidence_version=version["id"],
                    coverage={"rules": len(store.list_rules(version["id"])), "citations": len(citations)},
                )
        query_pipeline = req.app.state.query_pipeline
        tool = ComponentCheckTool(
            rag_service=query_pipeline.rag_service,
            generator=query_pipeline.generator,
            config=config,
        )
        result = tool.compare_with_template(
            doc_a=request.doc_a,
            doc_b=request.doc_b,
            params_template=request.params_template,
            strict_mode=request.strict_mode,
        )

        # ComponentCheckTool 返回的结论字段与 ComponentCompareResponse 一致
        return ComponentCompareResponse(
            conclusion=result.get("conclusion", ""),
            conclusion_detail=result.get("conclusion_detail", ""),
            params_comparison=result.get("params_comparison", []),
            failed_params=result.get("failed_params", []),
            directional_conclusions=result.get("directional_conclusions", {}),
            citations=result.get("citations", []),
            answer_status="ok" if not result.get("failed_params") else "insufficient_evidence",
            run_id=f"component-{uuid4().hex}",
        )
    except Exception as e:  # noqa: BLE001
        log.exception("/compare/component 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# POST /compare/visualize — EMC 表格可视化
# ============================================================

@router.post("/visualize", response_model=VisualizeEmcResponse)
async def visualize_emc(
    request: VisualizeEmcRequest,
    req: Request,
) -> VisualizeEmcResponse:
    """EMC 表格可视化

    将 RAG 检索到的 EMC 标准表格数据渲染为 ECharts 可视化 HTML 页面。
    支持：
        - 单文档可视化（传 doc_id）
        - 双文档叠加对比（传 doc_a + doc_b）
    """
    try:
        query_pipeline = req.app.state.query_pipeline
        tool = EmcVisualizeTool(rag_service=query_pipeline.rag_service)

        if request.doc_a and request.doc_b:
            # 双文档叠加对比
            result = tool.visualize_overlay(
                doc_a=request.doc_a,
                doc_b=request.doc_b,
                query=request.query,
                table_type=request.table_type,
                top_k=request.top_k * 2,
            )
        else:
            # 单文档可视化
            result = tool.visualize(
                query=request.query,
                doc_id=request.doc_id,
                top_k=request.top_k,
            )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # html_bundle 优先取 generate_html 的纯 HTML 字符串
        html_bundle = result.get("html") or result.get("bundle", "")
        if isinstance(html_bundle, dict):
            # bundle 可能是 dict 格式，转为字符串
            import json
            html_bundle = json.dumps(html_bundle, ensure_ascii=False)

        return VisualizeEmcResponse(
            html_bundle=str(html_bundle),
            table_type=result.get("type", "Unknown"),
            datasets=result.get("datasets", []),
            sources=result.get("sources", []),
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("/compare/visualize 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
