import fitz
import numpy as np
import cv2
from doctr.models import ocr_predictor
import re
import asyncio
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from config.patterns import PATRONES_INICIO, PATRON_DEFAULT
from app.core.config import get_settings


class DocumentClassifier:
    def __init__(self):
        self.settings = get_settings()
        self.ocr_engine = None
        self.executor = ThreadPoolExecutor(max_workers=self.settings.MAX_WORKERS)
    
    def initialize_ocr(self):
        """Inicializa el modelo docTR si no está inicializado."""
        if self.ocr_engine is None:
            self.ocr_engine = ocr_predictor(pretrained=True)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpia el texto manteniendo solo caracteres alfanuméricos y puntuación básica."""
        text_clean = re.sub(r'[^A-Z0-9\s:/.-]', '', text, flags=re.IGNORECASE)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        return text_clean.upper()
    
    @staticmethod
    def is_scanned(page: fitz.Page) -> bool:
        """Detecta si una página está escaneada."""
        native_text = page.get_text("text").strip()
        
        if len(native_text) < 10:
            image_list = page.get_images()
            if image_list:
                page_area = page.rect.width * page.rect.height
                image_area = 0
                
                for img in image_list:
                    try:
                        bbox = page.get_image_bbox(img[7])
                        image_area += bbox.width * bbox.height
                    except:
                        pass
                
                coverage = (image_area / page_area) * 100 if page_area > 0 else 0
                if coverage > 80:
                    return True
        
        return False
    
    def detect_orientation(self, page: fitz.Page) -> int:
        """Detecta la orientación de la página en grados."""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            if pix.n == 4:
                img_np = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            else:
                img_np = img_data
            
            img_normalized = img_np.astype(np.float32) / 255.0
            
            # NOTE: Se asume que initialize_ocr() es llamado por classify_document antes.
            # En caso de necesitarlo, descomentar self.initialize_ocr() aquí.
            result = self.ocr_engine([img_normalized])
            confidence_0 = 0
            words_0 = 0
            
            for page_result in result.pages:
                for block in page_result.blocks:
                    for line in block.lines:
                        for word in line.words:
                            confidence_0 += word.confidence
                            words_0 += 1
            
            if words_0 > 5 and (confidence_0 / words_0) > 0.5:
                return 0
            
            best_rotation = 0
            best_confidence = confidence_0 / max(words_0, 1)
            best_words = words_0
            
            for angle in [90, 180, 270]:
                if angle == 90:
                    img_rot = cv2.rotate(img_normalized, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img_rot = cv2.rotate(img_normalized, cv2.ROTATE_180)
                else:
                    img_rot = cv2.rotate(img_normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                result = self.ocr_engine([img_rot])
                confidence = 0
                words = 0
                
                for page_result in result.pages:
                    for block in page_result.blocks:
                        for line in block.lines:
                            for word in line.words:
                                confidence += word.confidence
                                words += 1
                
                avg_confidence = confidence / max(words, 1)
                
                if words > best_words or (words == best_words and avg_confidence > best_confidence):
                    best_rotation = angle
                    best_confidence = avg_confidence
                    best_words = words
            
            return best_rotation
        except Exception:
            return 0
    
    def correct_rotation(self, page: fitz.Page, angle: int):
        """Corrige la rotación de una página."""
        if angle != 0:
            page.set_rotation((page.rotation + angle) % 360)
    
    def extract_header_text(self, page: fitz.Page, mode: str) -> str:
        """Extrae texto del encabezado (30% superior) para clasificación."""
        header_rect = fitz.Rect(
            page.rect.x0,
            page.rect.y0,
            page.rect.x1,
            page.rect.y1 * self.settings.HEADER_PERCENTAGE
        )
        
        # CORRECCIÓN: Usar modos en inglés que coincidan con la entrada de la API
        if mode in ["NATIVE", "HYBRID"]:
            try:
                native_text = page.get_text("text", sort=True, clip=header_rect)
                clean = self.clean_text(native_text)
                if clean:
                    return clean
            except Exception:
                pass
        
        # CORRECCIÓN: Usar modos en inglés que coincidan con la entrada de la API
        if mode in ["OCR", "HYBRID"]:
            try:
                # Se asume que initialize_ocr() es llamado por classify_document antes
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(self.settings.OCR_DPI / 72, self.settings.OCR_DPI / 72),
                    clip=header_rect
                )
                
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                if pix.n == 4:
                    img_np = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:
                    img_np = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                else:
                    img_np = img_data
                
                img_normalized = img_np.astype(np.float32) / 255.0
                result = self.ocr_engine([img_normalized])
                
                text_ocr = ""
                for page_result in result.pages:
                    for block in page_result.blocks:
                        for line in block.lines:
                            for word in line.words:
                                text_ocr += word.value + " "
                
                return self.clean_text(text_ocr)
            except Exception:
                return ""
        
        return ""
    
    @staticmethod
    def classify_text(text: str) -> str:
        """Clasifica el texto según los patrones definidos."""
        for classification, patterns in PATRONES_INICIO.items():
            for pattern in patterns:
                if pattern.upper() in text:
                    return classification.upper()
        return ""
    
    def process_page(self, doc: fitz.Document, page_idx: int, mode: str) -> Dict:
        """Procesa una página individual."""
        try:
            page = doc.load_page(page_idx)
        except Exception:
            return {
                "page_index": page_idx,
                "classification": "",
                "is_scanned": False,
                "orientation": 0,
                "has_native_text": False,
                "error": "Failed to load page"
            }
        
        is_scanned = self.is_scanned(page)
        orientation = self.detect_orientation(page)
        
        if orientation != 0:
            self.correct_rotation(page, orientation)
        
        header_text = self.extract_header_text(page, mode)
        classification = self.classify_text(header_text)
        
        return {
            "page_index": page_idx,
            "classification": classification,
            "is_scanned": is_scanned,
            "orientation": orientation,
            "has_native_text": len(page.get_text("text").strip()) >= 10,
            "orientation_correct": orientation == 0
        }
    
    async def classify_document(self, pdf_bytes: bytes, mode: str = "HYBRID") -> List[Dict]:
        """
        Clasifica un documento PDF y devuelve la información de cada página.
        
        Args:
            pdf_bytes: Bytes del PDF
            mode: Modo de extracción (HYBRID, NATIVE, OCR)
            
        Returns:
            Lista de diccionarios con información de cada página
        """
        self.initialize_ocr()
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = doc.page_count
        
        # Convertir modo a mayúsculas una vez para consistencia
        mode_upper = mode.upper()
        
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.process_page,
                doc,
                i,
                mode_upper # Pasar el modo en mayúsculas
            )
            for i in range(num_pages)
        ]
        
        results = await asyncio.gather(*tasks)
        doc.close()
        
        return list(results)
    
    async def segment_document(self, pdf_bytes: bytes, mode: str = "HYBRID") -> List[Dict]:
        """
        Segmenta un documento PDF en múltiples documentos según clasificación.
        
        Args:
            pdf_bytes: Bytes del PDF
            mode: Modo de extracción
            
        Returns:
            Lista de segmentos con información de clasificación, rangos de páginas y calidad
        """
        page_results = await self.classify_document(pdf_bytes, mode)
        
        if not page_results:
            return []
        
        # Encontrar primer patrón válido
        first_pattern_idx = -1
        for i, result in enumerate(page_results):
            if result["classification"]:
                first_pattern_idx = i
                break
        
        if first_pattern_idx != -1:
            start_idx = first_pattern_idx
            current_classification = page_results[first_pattern_idx]["classification"]
        else:
            start_idx = 0
            current_classification = PATRON_DEFAULT
        
        segments = []
        
        for i in range(start_idx + 1, len(page_results)):
            found_classification = page_results[i]["classification"]
            
            if found_classification and found_classification != current_classification:
                segments.append({
                    "start_page": start_idx,
                    "end_page": i - 1,
                    "classification": current_classification,
                    "page_count": i - start_idx,
                    # ADICIÓN CLAVE: Incluir métricas de calidad de la página de inicio
                    "quality_metrics": {
                        "is_scanned": page_results[start_idx].get("is_scanned", False),
                        "orientation_degrees": page_results[start_idx].get("orientation", 0),
                        "orientation_correct": page_results[start_idx].get("orientation_correct", True),
                        "has_native_text": page_results[start_idx].get("has_native_text", False)
                    }
                })
                start_idx = i
                current_classification = found_classification
        
        # Añadir último segmento
        segments.append({
            "start_page": start_idx,
            "end_page": len(page_results) - 1,
            "classification": current_classification,
            "page_count": len(page_results) - start_idx,
            # ADICIÓN CLAVE: Incluir métricas de calidad de la página de inicio
            "quality_metrics": {
                "is_scanned": page_results[start_idx].get("is_scanned", False),
                "orientation_degrees": page_results[start_idx].get("orientation", 0),
                "orientation_correct": page_results[start_idx].get("orientation_correct", True),
                "has_native_text": page_results[start_idx].get("has_native_text", False)
            }
        })
        
        return segments