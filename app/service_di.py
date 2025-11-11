from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

class LocalDIService:
    def __init__(self, endpoint: str, api_key: str):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )

    def analyze(self, model_id: str, pdf_bytes: bytes):
        poller = self.client.begin_analyze_document(
            model_id,
            document=pdf_bytes
        )
        return poller.result()
