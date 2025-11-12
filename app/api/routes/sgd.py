from fastapi import APIRouter, HTTPException, status
from app.models.requests import SGDProcessRequest
from app.models.responses import ProcessResponse, DispatchInfoResponse
from app.services.sgd_service import SGDService
from app.services.document_processor import DocumentProcessor
from datetime import datetime
from app.utils.metrics import Timer


router = APIRouter(prefix="/sgd", tags=["SGD"])

sgd_service = SGDService()
processor = DocumentProcessor()


@router.get("/dispatch/{dispatch_code}/info", response_model=DispatchInfoResponse)
async def get_dispatch_info(dispatch_code: str):
    """
    Obtiene información detallada de un despacho sin procesar documentos.
    
    - **dispatch_code**: Código del despacho (interno o visible)
    """
    dispatch_info = await sgd_service.get_dispatch_info(dispatch_code)
    
    if not dispatch_info:
        raise HTTPException(
            status_code=404,
            detail=f"Dispatch not found: {dispatch_code}"
        )
    
    documents_list = dispatch_info.get("documentos", [])
    
    docs = []
    for doc in documents_list:
        docs.append({
            "name": doc.get("tipo", {}).get("nombre", "Unknown"),
            "status": doc.get("estado", "N/A"),
            "reception_date": doc.get("fecha_recepcion", "N/A")
        })
    
    return DispatchInfoResponse(
        success=True,
        dispatch_info=processor._build_dispatch_info(dispatch_info),
        documents_count=len(documents_list),
        documents_list=docs,
        timestamp=datetime.now()
    )


@router.post("/classify", response_model=ProcessResponse)
async def classify_dispatch(request: SGDProcessRequest):
    """
    Clasifica los documentos de un despacho sin extraer datos.
    
    - **dispatch_code**: Código del despacho
    """
    dispatch_info, documents = await sgd_service.fetch_dispatch_data(request.dispatch_code)
    
    if not dispatch_info:
        raise HTTPException(
            status_code=404,
            detail=f"Dispatch not found: {request.dispatch_code}"
        )
    
    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this dispatch"
        )
    
    # Procesar solo clasificación
    global_timing = {}
    processed_docs = []
    
    FIXED_EXTRACTION_MODE = "HYBRID"
    
    # Valores por defecto para satisfacer el modelo ProcessedDocument (Pydantic)
    DEFAULT_TIMING = {
        "fetch_time_ms": 0,
        "classification_time_ms": 0,
        "extraction_time_ms": 0,
        "total_time_ms": 0,
    }
    DEFAULT_QUALITY = {
        "is_scanned": False,
        "orientation_degrees": 0,
        "orientation_correct": True,
        "has_native_text": False
    }
    
    with Timer() as t:
        for doc_data in documents:
            pdf_bytes = sgd_service.decode_document(doc_data["documento"])
            
            segments = await processor.classifier.segment_document(
                pdf_bytes,
                FIXED_EXTRACTION_MODE
            )
            
            for segment in segments:
                processed_docs.append({
                    "document_id": str(doc_data.get("documento_id", "unknown")),
                    "document_name": doc_data.get("nombre_documento", "unknown.pdf"),
                    "document_type": segment["classification"],
                    "page_range": f"{segment['start_page'] + 1}-{segment['end_page'] + 1}",
                    "quality": DEFAULT_QUALITY,
                    "timing": DEFAULT_TIMING,
                    "extracted_data": None
                })
    
    global_timing["classification_time_ms"] = t.get_elapsed_ms()
    global_timing["total_time_ms"] = t.get_elapsed_ms()
    
    return ProcessResponse(
        success=True,
        dispatch_info=processor._build_dispatch_info(dispatch_info),
        total_documents=len(processed_docs),
        documents=processed_docs,
        timing=global_timing,
        alerts=[],
        timestamp=datetime.now()
    )


@router.post("/process", response_model=ProcessResponse)
async def process_dispatch(request: SGDProcessRequest):
    """
    Procesa completamente un despacho: clasifica documentos y extrae datos.
    
    - **dispatch_code**: Código del despacho
    - **use_cloud**: true para usar Azure DI cloud, false para local
    """
    result = await processor.process_sgd_dispatch(
        dispatch_code=request.dispatch_code,
        use_cloud=request.use_cloud
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Processing failed")
        )
    
    return ProcessResponse(
        success=result["success"],
        dispatch_info=result.get("dispatch_info"),
        total_documents=result["total_documents"],
        documents=result["documents"],
        timing=result["timing"],
        alerts=result["alerts"],
        timestamp=datetime.now()
    )