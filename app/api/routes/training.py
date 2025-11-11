from fastapi import APIRouter, HTTPException
from app.models.requests import TrainingRequest
from app.models.responses import TrainingResponse
from app.services.training_service import TrainingService
from datetime import datetime

router = APIRouter(prefix="/training", tags=["Training"])

training_service = TrainingService()


@router.get("/models")
async def list_models():
    """
    Lista todos los modelos disponibles en el contenedor Azure DI local.
    """
    models = await training_service.list_models()
    
    return {
        "success": True,
        "models": models,
        "count": len(models)
    }


@router.get("/folders")
async def list_training_folders():
    """
    Lista todas las carpetas de entrenamiento disponibles.
    Cada carpeta representa un modelo potencial.
    """
    folders = training_service.discover_training_folders()
    
    return {
        "success": True,
        "folders": folders,
        "count": len(folders)
    }


@router.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """
    Entrena modelos de Azure Document Intelligence.
    
    - **model_name**: Nombre específico del modelo a entrenar (opcional)
    - **force_retrain**: Forzar reentrenamiento si el modelo ya existe
    
    Si no se especifica model_name, entrena todos los modelos encontrados
    en la carpeta training_data.
    """
    if request.model_name:
        # Entrenar modelo específico
        folders = training_service.discover_training_folders()
        model_folder = next(
            (f for f in folders if f["model_name"] == request.model_name),
            None
        )
        
        if not model_folder:
            raise HTTPException(
                status_code=404,
                detail=f"Training folder not found for model: {request.model_name}"
            )
        
        model_id = training_service.get_model_version(request.model_name)
        
        success, result = await training_service.train_model(
            model_id=model_id,
            training_folder=model_folder["training_path"],
            description=f"Model for {request.model_name}"
        )
        
        if success:
            return TrainingResponse(
                success=True,
                models_trained=[model_id],
                models_failed=[],
                details={model_id: result},
                timestamp=datetime.now()
            )
        else:
            return TrainingResponse(
                success=False,
                models_trained=[],
                models_failed=[model_id],
                details={model_id: result},
                timestamp=datetime.now()
            )
    else:
        # Entrenar todos los modelos
        result = await training_service.train_all_models(
            force_retrain=request.force_retrain
        )
        
        return TrainingResponse(
            success=result["success"],
            models_trained=result["models_trained"],
            models_failed=result["models_failed"],
            details=result["details"],
            timestamp=datetime.now()
        )