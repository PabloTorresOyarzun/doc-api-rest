from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class DocumentQuality(BaseModel):
    is_scanned: bool = Field(..., description="Indica si el documento está escaneado")
    orientation_degrees: int = Field(..., description="Grados de rotación detectados (0, 90, 180, 270)")
    orientation_correct: bool = Field(..., description="Indica si la orientación es correcta")
    has_native_text: bool = Field(..., description="Indica si tiene texto nativo extraíble")


class DocumentTiming(BaseModel):
    fetch_time_ms: float = Field(..., description="Tiempo de obtención del documento en milisegundos")
    classification_time_ms: float = Field(..., description="Tiempo de clasificación en milisegundos")
    extraction_time_ms: float = Field(..., description="Tiempo de extracción de datos en milisegundos")
    total_time_ms: float = Field(..., description="Tiempo total del documento en milisegundos")


class DocumentAlert(BaseModel):
    type: str = Field(..., description="Tipo de alerta: quality, orientation, error")
    severity: str = Field(..., description="Severidad: info, warning, error")
    message: str = Field(..., description="Mensaje descriptivo de la alerta")
    document_name: Optional[str] = Field(None, description="Nombre del documento afectado")


class ProcessedDocument(BaseModel):
    document_id: str = Field(..., description="ID del documento")
    document_name: str = Field(..., description="Nombre del documento")
    document_type: str = Field(..., description="Tipo de documento clasificado")
    page_range: str = Field(..., description="Rango de páginas (ej: 1-3)")
    quality: DocumentQuality = Field(..., description="Métricas de calidad del documento")
    timing: DocumentTiming = Field(..., description="Métricas de tiempo del documento")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Datos extraídos del documento")


class DispatchInfo(BaseModel):
    dispatch_code: str = Field(..., description="Código visible del despacho")
    internal_id: str = Field(..., description="ID interno del despacho")
    client_name: str = Field(..., description="Nombre del cliente")
    status: str = Field(..., description="Estado del despacho")
    dispatch_type: str = Field(..., description="Tipo de despacho")
    assigned_users: List[Dict[str, str]] = Field(default_factory=list, description="Usuarios asignados")


class ProcessResponse(BaseModel):
    success: bool = Field(..., description="Indica si el proceso fue exitoso")
    dispatch_info: Optional[DispatchInfo] = Field(None, description="Información del despacho (solo para SGD)")
    total_documents: int = Field(..., description="Total de documentos procesados")
    documents: List[ProcessedDocument] = Field(..., description="Lista de documentos procesados")
    timing: Dict[str, float] = Field(..., description="Métricas de tiempo global en milisegundos")
    alerts: List[DocumentAlert] = Field(default_factory=list, description="Alertas de calidad y errores")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp del procesamiento")


class ClassifyResponse(BaseModel):
    success: bool = Field(..., description="Indica si la clasificación fue exitosa")
    total_documents: int = Field(..., description="Total de documentos identificados")
    documents: List[Dict[str, Any]] = Field(..., description="Lista de documentos clasificados")
    timing: Dict[str, float] = Field(..., description="Métricas de tiempo en milisegundos")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de la clasificación")


class TrainingResponse(BaseModel):
    success: bool = Field(..., description="Indica si el entrenamiento fue exitoso")
    models_trained: List[str] = Field(..., description="Lista de modelos entrenados")
    models_failed: List[str] = Field(default_factory=list, description="Lista de modelos que fallaron")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detalles del entrenamiento")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp del entrenamiento")


class DispatchInfoResponse(BaseModel):
    success: bool = Field(..., description="Indica si la consulta fue exitosa")
    dispatch_info: DispatchInfo = Field(..., description="Información del despacho")
    documents_count: int = Field(..., description="Cantidad de documentos asociados")
    documents_list: List[Dict[str, str]] = Field(..., description="Lista básica de documentos")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de la consulta")