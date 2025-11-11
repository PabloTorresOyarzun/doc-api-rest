import requests
import json
import base64
import zipfile
import io
import os
from pathlib import Path
import time

CONTAINER_ENDPOINT = "http://localhost:5000"
API_VERSION = "2023-07-31"

def create_training_zip(training_folder):
    """Crea un zip en memoria con los archivos de entrenamiento"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        training_path = Path(training_folder)
        
        for file_path in training_path.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(training_path)
                zip_file.write(file_path, arcname)
                print(f"Agregado: {arcname}")
    
    zip_buffer.seek(0)
    return base64.b64encode(zip_buffer.read()).decode('utf-8')

def train_model(model_id, training_folder, description=""):
    """Entrena un modelo en el contenedor local"""
    
    print(f"\n{'='*60}")
    print(f"Entrenando modelo: {model_id}")
    print(f"Carpeta: {training_folder}")
    print(f"{'='*60}\n")
    
    print("Creando archivo ZIP con datos de entrenamiento...")
    base64_zip = create_training_zip(training_folder)
    print(f"ZIP creado: {len(base64_zip)} caracteres base64\n")
    
    url = f"{CONTAINER_ENDPOINT}/formrecognizer/documentModels:build?api-version={API_VERSION}"
    
    payload = {
        "modelId": model_id,
        "description": description,
        "buildMode": "template",
        "base64Source": base64_zip
    }
    
    print("Enviando solicitud de entrenamiento...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 202:
        operation_location = response.headers.get('Operation-Location')
        print(f"Entrenamiento iniciado. Operation-Location: {operation_location}\n")
        
        print("Esperando a que termine el entrenamiento...")
        while True:
            status_response = requests.get(operation_location)
            status_data = status_response.json()
            
            status = status_data.get('status')
            print(f"Estado: {status}")
            
            if status == 'succeeded':
                print(f"\n✓ Modelo '{model_id}' entrenado exitosamente!\n")
                print(json.dumps(status_data.get('result', {}), indent=2))
                return True
            elif status == 'failed':
                print(f"\n✗ Error entrenando modelo '{model_id}':")
                print(json.dumps(status_data, indent=2))
                return False
            
            time.sleep(5)
    else:
        print(f"\n✗ Error iniciando entrenamiento:")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def list_models():
    """Lista modelos disponibles en el contenedor"""
    url = f"{CONTAINER_ENDPOINT}/formrecognizer/documentModels?api-version={API_VERSION}"
    response = requests.get(url)
    
    if response.status_code == 200:
        models = response.json().get('value', [])
        print(f"\n{'='*60}")
        print("MODELOS DISPONIBLES EN CONTENEDOR:")
        print(f"{'='*60}")
        for model in models:
            print(f"- {model.get('modelId')}: {model.get('description', 'Sin descripción')}")
        print(f"{'='*60}\n")
    else:
        print(f"Error listando modelos: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENTRENADOR DE MODELOS CUSTOM - CONTENEDOR LOCAL")
    print("="*60 + "\n")
    
    # Configuración de modelos a entrenar
    models_to_train = [
        {
            "model_id": "transport_01",
            "training_folder": "./training_data/transport",
            "description": "Extrae datos de documentos de transporte"
        },
        {
            "model_id": "inovice_01",
            "training_folder": "./training_data/invoice",
            "description": "Extrae datos de facturas aduaneras"
        }
    ]
    
    # Verificar que las carpetas existan
    for model_config in models_to_train:
        if not os.path.exists(model_config["training_folder"]):
            print(f"⚠ ADVERTENCIA: No existe la carpeta {model_config['training_folder']}")
            print(f"   Crea esta carpeta y pon los PDFs + archivos .labels.json allí\n")
    
    # Entrenar cada modelo
    for model_config in models_to_train:
        if os.path.exists(model_config["training_folder"]):
            train_model(
                model_config["model_id"],
                model_config["training_folder"],
                model_config["description"]
            )
            print("\n")
    
    # Listar modelos finales
    list_models()