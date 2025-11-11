from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Optional
from app.core.config import get_settings


class DocumentExtractor:
    def __init__(self, use_cloud: bool = False):
        self.settings = get_settings()
        self.use_cloud = use_cloud
        
        endpoint = self.settings.AZURE_ENDPOINT if use_cloud else self.settings.AZURE_LOCAL_ENDPOINT
        
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(self.settings.AZURE_KEY)
        )
    
    async def extract_data(self, model_id: str, pdf_bytes: bytes) -> Dict:
        """
        Extrae datos de un documento PDF usando un modelo de Azure DI.
        
        Args:
            model_id: ID del modelo a usar (ej: 'transport_01', 'invoice_01')
            pdf_bytes: Bytes del PDF
            
        Returns:
            Diccionario con los datos extraídos
        """
        try:
            poller = self.client.begin_analyze_document(
                model_id,
                document=pdf_bytes
            )
            result = poller.result()
            
            if not result.documents:
                return {}
            
            document = result.documents[0]
            extracted = {}
            
            for key, field in document.fields.items():
                if field.value is not None:
                    extracted[key] = {
                        "value": field.value,
                        "confidence": field.confidence
                    }
            
            return extracted
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_for_classification(self, classification: str) -> Optional[str]:
        """
        Mapea una clasificación de documento a un modelo de Azure DI.
        
        Args:
            classification: Clasificación del documento
            
        Returns:
            ID del modelo o None si no hay modelo disponible
        """
        model_mapping = {
            "FACTURA_COMERCIAL": "invoice_01",
            "DOCUMENTO_TRANSPORTE": "transport_01",
        }
        
        return model_mapping.get(classification)