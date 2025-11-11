# consultar_documentacion_con_fallback.py

import requests
import os
import json
import base64
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración de la API ---
BASE_URL = "https://backend.juanleon.cl"
ENDPOINT_DESPACHO = "/api/admin/despachos/{codigo}" 
ENDPOINT_DOCUMENTOS = "/api/admin/documentos64/despacho/{codigo_visible}"

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
# Código de despacho a consultar (el ID interno que funcionó en tu ejemplo)
CODIGO_DESPACHO_INTERNO = "54335" 
# Carpeta donde se guardarán los archivos
RUTA_GUARDADO = "documentos_despacho" 


# --- Funciones de Consulta ---

def consultar_despacho_detalle(codigo_interno: str, token: str):
    """
    Realiza una solicitud GET para obtener el detalle de un despacho específico (ID Interno).
    """
    if not token:
        print("Error: BEARER_TOKEN no encontrado en el archivo .env.")
        return None

    url = f"{BASE_URL}{ENDPOINT_DESPACHO.format(codigo=codigo_interno)}"
    headers = {
        "Accept": "application/json", 
        "Authorization": f"Bearer {token}" 
    }

    print(f"\n[1] INTENTO: Consultando detalle con código: {codigo_interno}")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Lanza excepción si el código de estado es 4xx/5xx

        return response.json()

    except requests.exceptions.RequestException as err:
        print(f"  -> FALLO en la solicitud de detalle: {err}")
    except Exception as e:
        print(f"  -> Ocurrió un error inesperado al consultar detalle: {e}")
        
    return None

def consultar_documentacion(codigo: str, token: str):
    """
    Realiza una solicitud GET para obtener la documentación (archivos en base64) 
    de un despacho específico (Código Visible o ID Interno, dependiendo del intento).
    """
    if not token:
        return None
    
    url = f"{BASE_URL}{ENDPOINT_DOCUMENTOS.format(codigo_visible=codigo)}"
    headers = {
        "Accept": "application/json", 
        "Authorization": f"Bearer {token}" 
    }
    
    print(f"Consultando documentación con código: {codigo}")
    print(f"URL: {url}")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        datos_json = response.json()
        
        # Devolvemos la lista de documentos, que está bajo la clave 'data'
        return datos_json.get("data", [])

    except requests.exceptions.RequestException as err:
        print(f"  -> FALLO en la solicitud de documentación: {err}")
    except Exception as e:
        print(f"  -> Ocurrió un error inesperado al consultar documentación: {e}")

    return None

# --- Función para Guardar PDF ---

def guardar_pdf_desde_base64(base64_data: str, nombre_archivo: str, ruta_guardado: str):
    """
    Decodifica una cadena Base64 que contiene un PDF y lo guarda en un archivo.
    """
    # Crear el directorio si no existe
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)

    # Eliminar el prefijo 'data:application/pdf;base64,' si existe
    if ',' in base64_data:
        _, base64_content = base64_data.split(',', 1)
    else:
        base64_content = base64_data
    
    ruta_completa = os.path.join(ruta_guardado, nombre_archivo)
    
    try:
        # Decodificar Base64 y escribir los datos binarios en un archivo ('wb')
        datos_decodificados = base64.b64decode(base64_content)
        
        with open(ruta_completa, 'wb') as f:
            f.write(datos_decodificados)
            
        return True
    
    except Exception as e:
        return False


# --- Ejecución principal ---

if __name__ == "__main__":
    
    if not BEARER_TOKEN:
        print("\n**¡ADVERTENCIA!** No se pudo obtener el BEARER_TOKEN. Revisa tu archivo .env.")
        exit()
    
    documentos_base64_list = None
    codigo_despacho_visible = None
    despacho_data = None
    
    # 1. INTENTO: Consultar Detalle (ID Interno)
    datos_despacho_detalle = consultar_despacho_detalle(CODIGO_DESPACHO_INTERNO, BEARER_TOKEN)

    if datos_despacho_detalle and "data" in datos_despacho_detalle:
        # ÉXITO en la consulta de detalle
        despacho_data = datos_despacho_detalle["data"]
        codigo_despacho_visible = despacho_data.get("codigo", "N/A")
        
        # Imprimir el output de detalle (que siempre quieres mantener)
        print("\n--- Respuesta de la API (Datos Extraídos) ---")
        estado = despacho_data.get("estado_despacho", "N/A")
        tipo = despacho_data.get("tipo_despacho", "N/A")
        cliente_nombre = despacho_data.get("cliente", {}).get("nombre", "N/A")
        documentos_list = despacho_data.get("documentos", [])
        usuarios_list = despacho_data.get("usuarios", [])

        print(f"Despacho Código N°: {codigo_despacho_visible} (ID interno: {despacho_data.get('id', 'N/A')})") 
        print(f"Cliente: {cliente_nombre}")
        print(f"Estado: **{estado}**")
        print(f"Tipo: **{tipo.upper()}**")
        
        print(f"\nDocumentos encontrados: {len(documentos_list)}")
        
        if documentos_list:
            for i, doc in enumerate(documentos_list, 1):
                nombre = doc.get("tipo", {}).get("nombre", "Sin nombre")
                recepcion = doc.get("fecha_recepcion", "N/A")
                estado_doc = doc.get("estado", "N/A")
                print(f"  {i}. {nombre} (Estado: {estado_doc}, Recibido: {recepcion})")
                
        print(f"\nUsuarios Asignados: {len(usuarios_list)}")
        pedidores = [u["name"] for u in usuarios_list if u.get("role_name") in ("pedidor_exportaciones", "pedidor")] 
        if pedidores:
            print(f"  - Pedidor: {', '.join(pedidores)}")
        jefes_op = [u["name"] for u in usuarios_list if u.get("role_name") == "jefe_operaciones"]
        if jefes_op:
            print(f"  - Jefe de Operaciones: {', '.join(jefes_op)}")

        # 2. INTENTO DE DOCUMENTACIÓN: Usar el código visible
        if codigo_despacho_visible and codigo_despacho_visible != "N/A":
            documentos_base64_list = consultar_documentacion(codigo_despacho_visible, BEARER_TOKEN)
    
    
    # 3. FALLBACK: Si no se encontró documentación con el código visible, intentar con el ID interno
    if not documentos_base64_list:
        if despacho_data:
            # Si se encontró el detalle pero falló la búsqueda de documentación con el código visible
            print("\nFALLO en la consulta de documentación con el código visible. Intentando con el ID interno...")
        else:
            # Si la consulta de detalle falló completamente
            print("\nFALLO en la consulta de detalle. Intentando consultar documentación directamente con el código original...")
        
        # 3. INTENTO DE DOCUMENTACIÓN (FALLBACK): Usar el ID interno
        documentos_base64_list = consultar_documentacion(CODIGO_DESPACHO_INTERNO, BEARER_TOKEN)

    
    # 4. PROCESAMIENTO FINAL: Guardar los archivos si se encontraron
    if documentos_base64_list and isinstance(documentos_base64_list, list):
        print("\n--- Guardando Documentación ---")
        print(f"Directorio de guardado: {RUTA_GUARDADO}")
        
        # Contador de archivos guardados con éxito
        archivos_guardados = 0
        
        for doc in documentos_base64_list:
            if isinstance(doc, dict):
                nombre_archivo = doc.get("nombre_documento", f"archivo_despacho_{doc.get('documento_id', 'desconocido')}.pdf")
                doc_base64 = doc.get("documento", "")
                
                # Llamar a la función de guardado
                if guardar_pdf_desde_base64(doc_base64, nombre_archivo, RUTA_GUARDADO):
                    archivos_guardados += 1
                    print(f"  -> OK: {nombre_archivo}")
                else:
                    print(f"  -> ERROR al guardar: {nombre_archivo}")
                    
        print(f"\nResumen de guardado: {archivos_guardados} de {len(documentos_base64_list)} archivos guardados.")

    else:
        # FALLO FINAL
        print("\n=======================================================")
        print("ERROR CRÍTICO: Fallo en todas las consultas.")
        print("No se pudo obtener el detalle del despacho ni la documentación con los códigos probados.")
        print(f"Códigos probados: ID Interno '{CODIGO_DESPACHO_INTERNO}' y su posible Código Visible ('{codigo_despacho_visible}' si se encontró).")
        print("Asegúrese de que el código de despacho y el BEARER_TOKEN sean válidos.")
        print("=======================================================")