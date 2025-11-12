import fitz
from typing import Dict, List, Optional
from app.services.sgd_service import SGDService
from app.services.document_classifier import DocumentClassifier
from app.services.document_extractor import DocumentExtractor
from app.models.responses import (
    ProcessedDocument, DocumentQuality, DocumentTiming,
    DocumentAlert, DispatchInfo
)
from app.utils.metrics import Timer


class DocumentProcessor:
    """Orquestador principal del procesamiento de documentos."""
    
    def __init__(self):
        self.sgd_service = SGDService()
        self.classifier = DocumentClassifier()
    
    async def process_sgd_dispatch(
        self,
        dispatch_code: str,
        extraction_mode: str,
        use_cloud: bool
    ) -> Dict:
        """
        Procesa un despacho completo desde SGD.
        
        Args:
            dispatch_code: Código del despacho
            extraction_mode: Modo de extracción de texto (HYBRID, NATIVE, OCR) 
            use_cloud: Usar Azure DI cloud
            
        Returns:
            Diccionario con toda la información del procesamiento
        """
        global_timing = {}
        alerts = []
        
        # Obtener información del despacho
        with Timer() as t:
            dispatch_info, documents = await self.sgd_service.fetch_dispatch_data(dispatch_code)
        
        global_timing["fetch_dispatch_info_ms"] = t.get_elapsed_ms()
        
        if not dispatch_info:
            return {
                "success": False,
                "error": "Failed to fetch dispatch information",
                "timing": global_timing
            }
        
        if not documents:
            return {
                "success": False,
                "error": "No documents found for dispatch",
                "dispatch_info": self._build_dispatch_info(dispatch_info),
                "timing": global_timing
            }
        
        # Procesar cada documento
        processed_docs = []
        
        with Timer() as t:
            for doc_data in documents:
                doc_result = await self._process_single_document(
                    doc_data,
                    extraction_mode,
                    use_cloud
                )
                processed_docs.append(doc_result["document"])
                alerts.extend(doc_result["alerts"])
        
        global_timing["process_all_documents_ms"] = t.get_elapsed_ms()
        global_timing["total_time_ms"] = sum(global_timing.values())
        
        return {
            "success": True,
            "dispatch_info": self._build_dispatch_info(dispatch_info),
            "total_documents": len(processed_docs),
            "documents": processed_docs,
            "timing": global_timing,
            "alerts": alerts
        }
    
    async def process_uploaded_document(
        self,
        pdf_bytes: bytes,
        filename: str,
        extraction_mode: str,
        use_cloud: bool
    ) -> Dict:
        """
        Procesa un documento subido directamente.
        
        Args:
            pdf_bytes: Bytes del PDF
            filename: Nombre del archivo
            extraction_mode: Modo de extracción (HYBRID, NATIVE, OCR)
            use_cloud: Usar Azure DI cloud
            
        Returns:
            Diccionario con la información del procesamiento
        """
        global_timing = {}
        alerts = []
        
        doc_data = {
            "nombre_documento": filename,
            "documento_id": "uploaded",
            "documento": pdf_bytes
        }
        
        with Timer() as t:
            doc_result = await self._process_single_document(
                doc_data,
                extraction_mode,
                use_cloud,
                is_base64=False
            )
        
        global_timing["process_document_ms"] = t.get_elapsed_ms()
        global_timing["total_time_ms"] = t.get_elapsed_ms()
        
        return {
            "success": True,
            "dispatch_info": None,
            "total_documents": 1,
            "documents": [doc_result["document"]],
            "timing": global_timing,
            "alerts": doc_result["alerts"]
        }
    
    async def classify_uploaded_document(
        self,
        pdf_bytes: bytes,
        filename: str,
        extraction_mode: str
    ) -> Dict:
        """
        Clasifica un documento subido sin extraer datos.
        
        Args:
            pdf_bytes: Bytes del PDF
            filename: Nombre del archivo
            extraction_mode: Modo de extracción (HYBRID, NATIVE, OCR)
            
        Returns:
            Diccionario con la clasificación
        """
        global_timing = {}
        
        with Timer() as t:
            segments = await self.classifier.segment_document(pdf_bytes, extraction_mode)
        
        global_timing["classification_time_ms"] = t.get_elapsed_ms()
        global_timing["total_time_ms"] = t.get_elapsed_ms()
        
        documents = []
        for segment in segments:
            documents.append({
                "document_type": segment["classification"],
                "page_range": f"{segment['start_page'] + 1}-{segment['end_page'] + 1}",
                "page_count": segment["page_count"],
                # ADICIÓN CLAVE: Incluir la calidad del documento (tomada de la primera página)
                "quality": segment.get("quality_metrics", {
                    "is_scanned": False,
                    "orientation_degrees": 0,
                    "orientation_correct": True,
                    "has_native_text": False
                })
            })
        
        return {
            "success": True,
            "total_documents": len(documents),
            "documents": documents,
            "timing": global_timing
        }
    
    async def _process_single_document(
        self,
        doc_data: Dict,
        extraction_mode: str,
        use_cloud: bool,
        is_base64: bool = True
    ) -> Dict:
        """Procesa un documento individual."""
        doc_timing = {}
        alerts = []
        
        doc_id = str(doc_data.get("documento_id", "unknown"))
        doc_name = doc_data.get("nombre_documento", "unknown.pdf")
        
        # Obtener bytes del documento
        if is_base64:
            with Timer() as t:
                pdf_bytes = self.sgd_service.decode_document(doc_data["documento"])
            doc_timing["fetch_time_ms"] = t.get_elapsed_ms()
        else:
            pdf_bytes = doc_data["documento"]
            doc_timing["fetch_time_ms"] = 0
        
        # Clasificar documento
        with Timer() as t:
            segments = await self.classifier.segment_document(pdf_bytes, extraction_mode)
        
        doc_timing["classification_time_ms"] = t.get_elapsed_ms()
        
        if not segments:
            alerts.append({
                "type": "error",
                "severity": "error",
                "message": "Document classification failed",
                "document_name": doc_name
            })
            return {
                "document": self._build_error_document(doc_id, doc_name, doc_timing),
                "alerts": alerts
            }
        
        # Tomar el primer segmento (documento principal)
        main_segment = segments[0]
        doc_type = main_segment["classification"]
        page_range = f"{main_segment['start_page'] + 1}-{main_segment['end_page'] + 1}"
        
        # Analizar calidad del documento
        quality = await self._analyze_document_quality(pdf_bytes, extraction_mode)
        
        # Generar alertas de calidad
        if quality.is_scanned:
            alerts.append({
                "type": "quality",
                "severity": "warning",
                "message": "Document is scanned. Digital original recommended.",
                "document_name": doc_name
            })
        
        if not quality.orientation_correct:
            alerts.append({
                "type": "orientation",
                "severity": "warning",
                "message": f"Document orientation incorrect ({quality.orientation_degrees}°). Please provide correctly oriented document.",
                "document_name": doc_name
            })
        
        # Extraer datos con Azure DI
        extracted_data = None
        extractor = DocumentExtractor(use_cloud=use_cloud)
        model_id = extractor.get_model_for_classification(doc_type)
        
        if model_id:
            with Timer() as t:
                extracted_data = await extractor.extract_data(model_id, pdf_bytes)
            doc_timing["extraction_time_ms"] = t.get_elapsed_ms()
        else:
            doc_timing["extraction_time_ms"] = 0
            alerts.append({
                "type": "info",
                "severity": "info",
                "message": f"No extraction model available for document type: {doc_type}",
                "document_name": doc_name
            })
        
        doc_timing["total_time_ms"] = sum(doc_timing.values())
        
        return {
            "document": {
                "document_id": doc_id,
                "document_name": doc_name,
                "document_type": doc_type,
                "page_range": page_range,
                "quality": quality.dict(),
                "timing": doc_timing,
                "extracted_data": extracted_data
            },
            "alerts": alerts
        }
    
    async def _analyze_document_quality(self, pdf_bytes: bytes, mode: str) -> DocumentQuality:
        """Analiza la calidad de un documento."""
        page_results = await self.classifier.classify_document(pdf_bytes, mode)
        
        if not page_results:
            return DocumentQuality(
                is_scanned=False,
                orientation_degrees=0,
                orientation_correct=True,
                has_native_text=False
            )
        
        # Tomar la primera página como referencia
        first_page = page_results[0]
        
        return DocumentQuality(
            is_scanned=first_page.get("is_scanned", False),
            orientation_degrees=first_page.get("orientation", 0),
            orientation_correct=first_page.get("orientation_correct", True),
            has_native_text=first_page.get("has_native_text", False)
        )
    
    @staticmethod
    def _build_dispatch_info(dispatch_data: Dict) -> DispatchInfo:
        """Construye objeto DispatchInfo desde los datos de SGD."""
        usuarios = dispatch_data.get("usuarios", [])
        assigned_users = []
        
        for user in usuarios:
            assigned_users.append({
                "name": user.get("name", ""),
                "role": user.get("role_name", "")
            })
        
        return DispatchInfo(
            dispatch_code=dispatch_data.get("codigo", "N/A"),
            internal_id=str(dispatch_data.get("id", "N/A")),
            client_name=dispatch_data.get("cliente", {}).get("nombre", "N/A"),
            status=dispatch_data.get("estado_despacho", "N/A"),
            dispatch_type=dispatch_data.get("tipo_despacho", "N/A"),
            assigned_users=assigned_users
        )
    
    @staticmethod
    def _build_error_document(doc_id: str, doc_name: str, timing: Dict) -> ProcessedDocument:
        """Construye un documento de error."""
        # Asegurar que timing tenga todos los campos requeridos
        complete_timing = {
            "fetch_time_ms": timing.get("fetch_time_ms", 0),
            "classification_time_ms": timing.get("classification_time_ms", 0),
            "extraction_time_ms": 0,
            "total_time_ms": timing.get("total_time_ms", sum(timing.values()) if timing else 0)
        }
        
        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "document_type": "ERROR",
            "page_range": "N/A",
            "quality": {
                "is_scanned": False,
                "orientation_degrees": 0,
                "orientation_correct": True,
                "has_native_text": False
            },
            "timing": complete_timing,
            "extracted_data": None
        }