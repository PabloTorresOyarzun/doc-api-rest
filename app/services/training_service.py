import httpx
import base64
import zipfile
import io
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from app.core.config import get_settings


class TrainingService:
    def __init__(self):
        self.settings = get_settings()
        self.container_endpoint = self.settings.AZURE_LOCAL_ENDPOINT
        self.api_version = "2023-07-31"
    
    @staticmethod
    def create_training_zip(training_folder: Path) -> str:
        """Crea un zip en memoria con los archivos de entrenamiento."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in training_folder.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(training_folder)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        return base64.b64encode(zip_buffer.read()).decode('utf-8')
    
    def discover_training_folders(self) -> List[Dict[str, str]]:
        """
        Descubre automáticamente las carpetas de entrenamiento.
        
        Returns:
            Lista de diccionarios con model_name y training_path
        """
        training_base = Path(self.settings.TRAINING_DATA_DIR)
        
        if not training_base.exists():
            return []
        
        models = []
        for folder in training_base.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                models.append({
                    "model_name": folder.name,
                    "training_path": str(folder)
                })
        
        return models
    
    def get_model_version(self, model_name: str) -> str:
        """
        Determina la siguiente versión disponible para un modelo.
        
        Args:
            model_name: Nombre base del modelo
            
        Returns:
            Nombre del modelo con versión (ej: 'invoice_01', 'invoice_02')
        """
        # Por simplicidad, inicia en 01 y podría verificar modelos existentes
        return f"{model_name}_01"
    
    async def list_models(self) -> List[Dict]:
        """Lista modelos disponibles en el contenedor."""
        url = f"{self.container_endpoint}/formrecognizer/documentModels?api-version={self.api_version}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                return response.json().get('value', [])
            except httpx.HTTPError:
                return []
    
    async def train_model(self, model_id: str, training_folder: str, description: str = "") -> Tuple[bool, Dict]:
        """
        Entrena un modelo en el contenedor local.
        
        Args:
            model_id: ID del modelo a entrenar
            training_folder: Ruta de la carpeta con datos de entrenamiento
            description: Descripción del modelo
            
        Returns:
            Tupla (success, details)
        """
        folder_path = Path(training_folder)
        
        if not folder_path.exists():
            return False, {"error": f"Training folder not found: {training_folder}"}
        
        try:
            base64_zip = self.create_training_zip(folder_path)
        except Exception as e:
            return False, {"error": f"Failed to create training zip: {str(e)}"}
        
        url = f"{self.container_endpoint}/formrecognizer/documentModels:build?api-version={self.api_version}"
        
        payload = {
            "modelId": model_id,
            "description": description,
            "buildMode": "template",
            "base64Source": base64_zip
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(url, json=payload)
                
                if response.status_code != 202:
                    return False, {
                        "error": f"Training initiation failed with status {response.status_code}",
                        "response": response.text
                    }
                
                operation_location = response.headers.get('Operation-Location')
                
                if not operation_location:
                    return False, {"error": "No Operation-Location header returned"}
                
                # Polling del estado
                while True:
                    status_response = await client.get(operation_location)
                    status_data = status_response.json()
                    
                    status = status_data.get('status')
                    
                    if status == 'succeeded':
                        return True, status_data.get('result', {})
                    elif status == 'failed':
                        return False, status_data
                    
                    await asyncio.sleep(5)
                    
            except httpx.HTTPError as e:
                return False, {"error": f"HTTP error: {str(e)}"}
            except Exception as e:
                return False, {"error": f"Unexpected error: {str(e)}"}
    
    async def train_all_models(self, force_retrain: bool = False) -> Dict:
        """
        Entrena todos los modelos encontrados en la carpeta de entrenamiento.
        
        Args:
            force_retrain: Forzar reentrenamiento de modelos existentes
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        models_to_train = self.discover_training_folders()
        
        if not models_to_train:
            return {
                "success": False,
                "message": "No training folders found",
                "models_trained": [],
                "models_failed": []
            }
        
        existing_models = await self.list_models()
        existing_model_ids = [m.get('modelId', '') for m in existing_models]
        
        models_trained = []
        models_failed = []
        details = {}
        
        for model_config in models_to_train:
            model_name = model_config["model_name"]
            model_id = self.get_model_version(model_name)
            
            if model_id in existing_model_ids and not force_retrain:
                details[model_id] = {"status": "skipped", "reason": "Model already exists"}
                continue
            
            success, result = await self.train_model(
                model_id,
                model_config["training_path"],
                f"Model for {model_name}"
            )
            
            if success:
                models_trained.append(model_id)
                details[model_id] = {"status": "success", "result": result}
            else:
                models_failed.append(model_id)
                details[model_id] = {"status": "failed", "error": result}
        
        return {
            "success": len(models_failed) == 0,
            "models_trained": models_trained,
            "models_failed": models_failed,
            "details": details
        }


import asyncio