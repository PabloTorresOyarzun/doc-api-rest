from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.responses import ProcessResponse, ClassifyResponse
from app.services.document_processor import DocumentProcessor
from datetime import datetime

router = APIRouter(prefix="/documents", tags=["Documents"])

processor = DocumentProcessor()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_document(
    file: UploadFile = File(...)
):
    """
    Clasifica un documento PDF cargado sin extraer datos.
    
    - **file**: Archivo PDF
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    pdf_bytes = await file.read()
    
    result = await processor.classify_uploaded_document(
        pdf_bytes=pdf_bytes,
        filename=file.filename,
    )
    
    return ClassifyResponse(
        success=result["success"],
        total_documents=result["total_documents"],
        documents=result["documents"],
        timing=result["timing"],
        timestamp=datetime.now()
    )


@router.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    use_cloud: bool = Form(default=False)
):
    """
    Procesa completamente un documento PDF: clasifica y extrae datos.
    
    - **file**: Archivo PDF
    - **use_cloud**: true para usar Azure DI cloud, false para local
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    pdf_bytes = await file.read()
    
    result = await processor.process_uploaded_document(
        pdf_bytes=pdf_bytes,
        filename=file.filename,
        use_cloud=use_cloud
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