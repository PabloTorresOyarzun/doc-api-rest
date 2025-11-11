from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import sgd, documents, training

settings = get_settings()

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
API para procesamiento automático de documentos aduaneros.

## Características

- **SGD Integration**: Consulta y procesamiento de despachos
- **Document Classification**: Identificación automática de tipos de documentos
- **Data Extraction**: Extracción de datos usando Azure Document Intelligence
- **Model Training**: Entrenamiento de modelos personalizados
- **Quality Metrics**: Análisis de calidad y orientación de documentos
- **Performance Tracking**: Métricas de tiempo para cada operación

## Flujo de Trabajo

1. Obtener despacho desde SGD o cargar documento
2. Clasificar documentos (identifica tipos)
3. Extraer datos con modelos entrenados
4. Recibir respuesta con métricas y alertas de calidad
"""
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(sgd.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Customs Document Processor API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.API_VERSION
    }