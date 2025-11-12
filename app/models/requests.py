from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ExtractionMode(str, Enum):
    HYBRID = "HYBRID"
    NATIVE = "NATIVE"
    OCR = "OCR"


class SGDProcessRequest(BaseModel):
    dispatch_code: str = Field(..., description="Código de despacho SGD")
    use_cloud: bool = Field(default=False, description="Usar Azure DI cloud en lugar de local")


class DocumentClassifyRequest(BaseModel):
    pass


class DocumentProcessRequest(BaseModel):
    use_cloud: bool = Field(default=False, description="Usar Azure DI cloud en lugar de local")


class TrainingRequest(BaseModel):
    model_name: Optional[str] = Field(None, description="Nombre del modelo específico a entrenar. Si no se proporciona, entrena todos")
    force_retrain: bool = Field(default=False, description="Forzar reentrenamiento si el modelo ya existe")