import httpx
import base64
from typing import Dict, List, Optional, Tuple
from app.core.config import get_settings


class SGDService:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.SGD_BASE_URL
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.settings.SGD_BEARER_TOKEN}"
        }
    
    async def get_dispatch_info(self, dispatch_code: str) -> Optional[Dict]:
        """
        Obtiene la información detallada de un despacho.
        
        Args:
            dispatch_code: Código del despacho (interno o visible)
            
        Returns:
            Diccionario con la información del despacho o None si falla
        """
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/api/admin/despachos/{dispatch_code}"
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                return response.json().get("data")
            except httpx.HTTPError:
                return None
    
    async def get_dispatch_documents(self, dispatch_code: str) -> Optional[List[Dict]]:
        """
        Obtiene los documentos (en base64) asociados a un despacho.
        
        Args:
            dispatch_code: Código del despacho (interno o visible)
            
        Returns:
            Lista de documentos en base64 o None si falla
        """
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/api/admin/documentos64/despacho/{dispatch_code}"
                response = await client.get(url, headers=self.headers, timeout=60.0)
                response.raise_for_status()
                return response.json().get("data", [])
            except httpx.HTTPError:
                return None
    
    async def fetch_dispatch_data(self, dispatch_code: str) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Obtiene tanto la información del despacho como sus documentos.
        Intenta con el código proporcionado y, si falla en documentos, usa el código visible.
        
        Args:
            dispatch_code: Código del despacho
            
        Returns:
            Tupla (dispatch_info, documents_list)
        """
        dispatch_info = await self.get_dispatch_info(dispatch_code)
        
        if not dispatch_info:
            return None, None
        
        visible_code = dispatch_info.get("codigo")
        documents = await self.get_dispatch_documents(visible_code)
        
        if not documents:
            documents = await self.get_dispatch_documents(dispatch_code)
        
        return dispatch_info, documents
    
    @staticmethod
    def decode_document(base64_data: str) -> bytes:
        """
        Decodifica un documento en base64 a bytes.
        
        Args:
            base64_data: String en base64 del documento
            
        Returns:
            Bytes del documento
        """
        if ',' in base64_data:
            _, base64_content = base64_data.split(',', 1)
        else:
            base64_content = base64_data
        
        return base64.b64decode(base64_content)