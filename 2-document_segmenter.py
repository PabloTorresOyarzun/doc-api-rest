import fitz
import os
import numpy as np
import cv2
from doctr.models import ocr_predictor
import re
from concurrent.futures import ThreadPoolExecutor # Para paralelización

# Importar la configuración de patrones
from configuracion_patrones import PATRONES_INICIO, PATRON_DEFAULT 

# --- CONFIGURACION FIJA ---
PDF_ENTRADA = "docs/archivo_consolidado.pdf"
CARPETA_SALIDA = "documentos_separados_clasificados"
MAX_WORKERS = 12 # Número de procesos para el procesamiento de páginas
# ---------------------

def inicializar_ocr():
    """
    Inicializa el modelo docTR.
    """
    ocr = ocr_predictor(pretrained=True)
    return ocr

def detectar_si_esta_escaneada(pagina: fitz.Page) -> bool:
    """
    Detecta si una pagina esta escaneada verificando texto nativo y cobertura de imagenes.
    """
    # Verificar texto nativo
    texto_nativo = pagina.get_text("text").strip()
    
    if len(texto_nativo) < 10:
        # Verificar cobertura de imagenes
        lista_imagenes = pagina.get_images()
        if lista_imagenes:
            area_pagina = pagina.rect.width * pagina.rect.height
            area_imagenes = 0
            
            for img in lista_imagenes:
                try:
                    # Usar el índice 7 para el bbox como en la versión original
                    bbox = pagina.get_image_bbox(img[7]) 
                    area_imagenes += bbox.width * bbox.height
                except:
                    pass
            
            cobertura = (area_imagenes / area_pagina) * 100 if area_pagina > 0 else 0
            
            if cobertura > 80:
                return True
    
    return False

def detectar_orientacion(pagina: fitz.Page, ocr_engine) -> int:
    """
    Detecta la orientacion de la pagina en grados (0, 90, 180, 270).
    Retorna el angulo de rotacion necesario para corregir.
    """
    try:
        # Renderizar a imagen con menor resolucion para velocidad (150 DPI)
        pix = pagina.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
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
        pix = None
        
        # Probar orientacion actual (0 grados)
        resultado = ocr_engine([img_normalized])
        confianza_0 = 0
        palabras_0 = 0
        
        for page in resultado.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        confianza_0 += word.confidence
                        palabras_0 += 1
        
        if palabras_0 > 5 and (confianza_0 / palabras_0) > 0.5:
            return 0
            
        # Probar rotaciones
        mejor_rotacion = 0
        mejor_confianza = confianza_0 / max(palabras_0, 1)
        mejor_palabras = palabras_0
        
        for angulo in [90, 180, 270]:
            if angulo == 90:
                img_rot = cv2.rotate(img_normalized, cv2.ROTATE_90_CLOCKWISE)
            elif angulo == 180:
                img_rot = cv2.rotate(img_normalized, cv2.ROTATE_180)
            else:
                img_rot = cv2.rotate(img_normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            resultado = ocr_engine([img_rot])
            confianza = 0
            palabras = 0
            
            for page in resultado.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            confianza += word.confidence
                            palabras += 1
            
            confianza_promedio = confianza / max(palabras, 1)
            
            if palabras > mejor_palabras or (palabras == mejor_palabras and confianza_promedio > mejor_confianza):
                mejor_rotacion = angulo
                mejor_confianza = confianza_promedio
                mejor_palabras = palabras
        
        return mejor_rotacion
        
    except Exception as e:
        print(f"  ERROR en deteccion de orientacion: {e}")
        return 0

def corregir_rotacion_pagina(pagina: fitz.Page, angulo: int):
    """
    Corrige la rotacion de una pagina.
    """
    if angulo != 0:
        pagina.set_rotation((pagina.rotation + angulo) % 360)
        
def limpiar_texto(texto: str) -> str:
    """
    Limpia el texto (principalmente de OCR) manteniendo solo 
    caracteres alfanuméricos, espacios, y signos comunes de puntuación.
    """
    # Elimina cualquier carácter que no sea una letra, número, espacio, o ':', '-', '/', '.'
    texto_limpio = re.sub(r'[^A-Z0-9\s:/.-]', '', texto, flags=re.IGNORECASE)
    # Normaliza múltiples espacios
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio.upper()

# --- NUEVA FUNCIÓN PARA EXTRACCIÓN LIMITADA (30% SUPERIOR) ---
def extraer_texto_de_encabezado(pagina: fitz.Page, ocr_engine, modo_extraccion: str) -> str:
    """
    Extrae texto ÚNICAMENTE del 30% superior de la página (encabezado) 
    para fines de CLASIFICACIÓN.
    """
    
    # --- DEFINICIÓN DEL ÁREA DEL ENCABEZADO (30% superior) ---
    rect_header = fitz.Rect(
        pagina.rect.x0, 
        pagina.rect.y0, 
        pagina.rect.x1, 
        pagina.rect.y1 * 0.30 
    )
    # ----------------------------------------------------------
    
    # 1. Intento de extraccion nativa limitada
    if modo_extraccion in ["NATIVO", "HIBRIDO"]:
        try:
            texto_nativo_raw = pagina.get_text("text", sort=True, clip=rect_header) 
            texto_nativo = limpiar_texto(texto_nativo_raw) 
            
            if texto_nativo:
                return texto_nativo
        except Exception:
            pass
        
    # 2. Usar OCR limitado
    if modo_extraccion in ["OCR", "HIBRIDO"]:
        try:
            # Renderizar SÓLO el área del encabezado a 300 DPI
            pix = pagina.get_pixmap(
                matrix=fitz.Matrix(300 / 72, 300 / 72), 
                clip=rect_header 
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
            pix = None

            # Ejecutar OCR con docTR
            resultado = ocr_engine([img_normalized])
            
            # Extraer texto del resultado
            texto_ocr_raw = ""
            for page in resultado.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            texto_ocr_raw += word.value + " "
            
            return limpiar_texto(texto_ocr_raw) 
        except Exception:
            return ""
            
    return ""
# --------------------------------------------------------------------------

def extraer_texto_de_pagina(pagina: fitz.Page, ocr_engine, modo_extraccion: str) -> str:
    """
    Extrae texto usando el modo especificado: HIBRIDO, NATIVO o OCR.
    Esta función se encarga del pre-procesamiento (detección de orientación y corrección).
    """
    
    # --- PASO DE PRE-PROCESAMIENTO FORZADO ---
    try: 
        # Deteccion de escaneado (informativo)
        is_scanned = detectar_si_esta_escaneada(pagina)
        if is_scanned:
            print("  -> Pagina escaneada detectada.")
        else:
            print("  -> Pagina con texto nativo detectada.")
        
        # Deteccion y correccion de orientacion (ejecución costosa)
        angulo_correccion = detectar_orientacion(pagina, ocr_engine)
        if angulo_correccion != 0:
            print(f"  -> Orientacion incorrecta detectada. Corrigiendo {angulo_correccion} grados.")
            corregir_rotacion_pagina(pagina, angulo_correccion)
        else:
            print("  -> Orientacion correcta.")

    except Exception as e:
        print(f"  ERROR CRÍTICO al pre-procesar la Pag. {pagina.number + 1}: Es probable que la página esté corrupta o malformada. {e}")
        return "" 
        
    # ----------------------------------------
    
    # 1. Intento de extraccion nativa (Página COMPLETA)
    texto_nativo = ""
    if modo_extraccion in ["NATIVO", "HIBRIDO"]:
        try:
            # Obtener el texto nativo DESPUÉS de aplicar la rotación
            texto_nativo_raw = pagina.get_text("text", sort=True) 
            texto_nativo = limpiar_texto(texto_nativo_raw) # <--- Limpieza de Texto
            
            if texto_nativo:
                print("  -> Usando extraccion de texto nativo (PyMuPDF).")
                return texto_nativo
        except Exception:
            if modo_extraccion == "NATIVO":
                 print("  -> Advertencia: Fallo la extraccion nativa.")
                 return ""
            pass
        
        if modo_extraccion == "NATIVO":
             print("  -> Advertencia: No se encontro texto nativo.")
             return ""
             
    # 2. Usar OCR (Si el modo es OCR o HIBRIDO falló en el nativo) (Página COMPLETA)
    if modo_extraccion in ["OCR", "HIBRIDO"]:
        
        if modo_extraccion == "HIBRIDO":
             print("  -> Usando docTR (Texto nativo no encontrado o fallo).")
        elif modo_extraccion == "OCR":
             print("  -> Usando docTR (Modo forzado).")

        try:
            # Renderizar la pagina a imagen a 300 DPI
            pix = pagina.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # ... (conversión a RGB y normalización) ...
            if pix.n == 4:
                img_np = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 3:
                img_np = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            else:
                img_np = img_data

            img_normalized = img_np.astype(np.float32) / 255.0
            pix = None

            # Ejecutar OCR con docTR
            resultado = ocr_engine([img_normalized])
            
            # Extraer texto del resultado
            texto_ocr_raw = ""
            for page in resultado.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            texto_ocr_raw += word.value + " "
            
            texto_final_ocr = limpiar_texto(texto_ocr_raw) # <--- Limpieza de Texto
            print(f"    [OCR RESULTADO LIMPIO]: '{texto_final_ocr}'")
            
            return texto_final_ocr
        except Exception as e:
            print(f"  ERROR critico en OCR en Pag. {pagina.number + 1}: {e}")
            return ""
            
    return ""

def clasificar_pagina(texto_pagina: str) -> str:
    """
    Busca CUALQUIERA de los patrones en la lista asociada a una clasificación 
    en el texto de la página y devuelve la CLASIFICACIÓN (la clave del diccionario).
    """
    for clasificacion, patrones_lista in PATRONES_INICIO.items():
        for patron_valor in patrones_lista:
            # Ahora, 'texto_pagina' es solo el encabezado, lo que mejora la precisión.
            if patron_valor.upper() in texto_pagina: 
                return clasificacion.upper()
    return ""

def procesar_pagina(i, doc_original, ocr_engine, modo_extraccion):
    """
    Función auxiliar para procesar una página, utilizada por ThreadPoolExecutor.
    Retorna (indice_pagina, clasificacion_encontrada).
    """
    
    try:
        pagina = doc_original.load_page(i) 
    except Exception as e:
        print(f"  ERROR: No se pudo cargar la Pagina {i + 1}. Omitiendo. {e}")
        return i, "" # Retorna vacío si falla la carga

    # 1. Extracción COMPLETA (maneja pre-procesamiento, rotación y logging)
    # Se extrae la página completa (o lo que se pueda) para el logging del OCR y la corrección de rotación.
    texto_pagina_completo = extraer_texto_de_pagina(pagina, ocr_engine, modo_extraccion)
    
    if not texto_pagina_completo:
        # Si la extracción completa falla (página corrupta), no podemos clasificar.
        print(f"--- PAGINA {i + 1} (Finalizado) --- ERROR DE EXTRACCIÓN/CONTINUACION")
        return i, ""
    
    # 2. Extracción SÓLO DE ENCABEZADO y Clasificación (usa el 30% superior)
    # Usa la página ya corregida en el paso 1.
    texto_encabezado = extraer_texto_de_encabezado(pagina, ocr_engine, modo_extraccion)
    clasificacion_encontrada = clasificar_pagina(texto_encabezado)

    # Imprimir resultado en el proceso:
    if clasificacion_encontrada:
        print(f"--- PAGINA {i + 1} (Finalizado) --- CLASIFICACION: **{clasificacion_encontrada}**")
    else:
        print(f"--- PAGINA {i + 1} (Finalizado) --- CONTINUACION (No hay patrón)")
    
    return i, clasificacion_encontrada

def dividir_pdf_por_contenido(archivo_pdf: str, carpeta_salida: str, modo_extraccion: str):
    """
    Segmenta el PDF buscando CUALQUIERA de los patrones de inicio y CLASIFICA 
    el documento por el patrón encontrado, utilizando paralelismo.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta de salida creada: {carpeta_salida}")

    try:
        ocr = inicializar_ocr()
    except Exception as e:
        print(f"\nERROR al inicializar docTR: {e}")
        return

    # --- Manejo de Archivos Protegidos / Corruptos ---
    doc_original = None
    try:
        doc_original = fitz.open(archivo_pdf)
        
        if doc_original.needs_pass:
             print(f"\nERROR CRÍTICO: El archivo '{archivo_pdf}' está **protegido con contraseña** y no se pudo abrir. Proceso detenido.")
             return
             
        if doc_original.is_repaired:
             print(f"\nADVERTENCIA: El archivo '{archivo_pdf}' estaba **corrupto y fue reparado** por PyMuPDF. Verifique los resultados.")
             
    except Exception as e:
        print(f"\nERROR CRÍTICO: No se pudo abrir el archivo PDF '{archivo_pdf}'. Podría ser inválido, corrupto o faltante. {e}")
        return
    # ------------------------------------------------
        
    num_paginas = doc_original.page_count
    print(f"\nProcesando PDF: '{archivo_pdf}' con {num_paginas} paginas (Paralelizando con {MAX_WORKERS} workers)...")

    # --- Ejecución Paralela del Procesamiento de Páginas ---
    resultados_paginas = [None] * num_paginas
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(procesar_pagina, i, doc_original, ocr, modo_extraccion): i
            for i in range(num_paginas)
        }
        
        for future in future_to_page:
            try:
                i, clasificacion = future.result() 
                resultados_paginas[i] = clasificacion
            except Exception as exc:
                print(f"ERROR en el procesamiento paralelo de una página: {exc}")
                resultados_paginas[future_to_page[future]] = "" # Poner vacío si falla

    # --- Lógica de Segmentación SECUENCIAL usando los resultados ---
    
    limites_documentos = []
    inicio_doc_actual = 0
    clasificacion_doc_actual = "DESCONOCIDO" 

    # 1. Ajuste de Clasificación Inicial (Mejora 3 - Corregida)
    # Busca el primer patrón válido en todo el documento.
    
    primer_patron_i = -1
    for i in range(num_paginas):
        if resultados_paginas[i]: # Si no es cadena vacía ("")
            primer_patron_i = i
            break
            
    if primer_patron_i != -1:
        inicio_doc_actual = primer_patron_i
        clasificacion_doc_actual = resultados_paginas[primer_patron_i]
        print(f"\n-> CLASIFICACION INICIAL: Documento comienza como: **{clasificacion_doc_actual}** en Pag. {inicio_doc_actual + 1}.")
    else:
        # Si no hay ningún patrón en todo el PDF, se usa el default para clasificarlo todo
        clasificacion_doc_actual = PATRON_DEFAULT
        print(f"\n-> CLASIFICACION INICIAL POR DEFECTO: No se encontraron patrones. Usando: **{clasificacion_doc_actual}**.")


    # 2. Segmentación del Resto del Documento
    for i in range(inicio_doc_actual + 1, num_paginas):
        clasificacion_encontrada = resultados_paginas[i]
        
        if clasificacion_encontrada: # Si encontramos un patrón (no es cadena vacía "")
            
            # Si el patrón encontrado es diferente al actual, es el inicio de un nuevo documento
            if clasificacion_encontrada != clasificacion_doc_actual:
                 # Finaliza el documento anterior
                limites_documentos.append((inicio_doc_actual, i - 1, clasificacion_doc_actual))
                print(f"  -> INICIO DETECTADO: Doc. anterior ({clasificacion_doc_actual}) termina en Pag. {i}. ")
                
                # Inicia el nuevo documento
                inicio_doc_actual = i
                clasificacion_doc_actual = clasificacion_encontrada
                print(f"  -> CLASIFICACION: Nuevo Doc. comienza aqui como: **{clasificacion_doc_actual}**.")
            else:
                 # El mismo patrón se repite (ej. encabezados), continúa el documento actual
                 print(f"  -> Pagina {i+1}: Patrón '{clasificacion_encontrada}' repetido. Continúa el documento.")

        else: # clasificacion_encontrada es "" (No tiene patrón)
             print(f"  -> Pagina {i+1}: No se encontró ningún patrón de inicio. Continúa con la clasificación: {clasificacion_doc_actual}.")

    # Añadir el último documento
    if inicio_doc_actual < num_paginas:
        limites_documentos.append((inicio_doc_actual, num_paginas - 1, clasificacion_doc_actual))
        print(f"\n  -> Ultimo documento ({clasificacion_doc_actual}): Paginas {inicio_doc_actual + 1} a {num_paginas}")

    print("\n==================================")
    print("--- INICIANDO DIVISION Y GUARDADO ---")
    print("==================================")
    
    contador_clasificacion = {} 
    
    for inicio, fin, clasificacion in limites_documentos:
        if clasificacion not in contador_clasificacion:
            contador_clasificacion[clasificacion] = 1
        else:
            contador_clasificacion[clasificacion] += 1
            
        indice = contador_clasificacion[clasificacion]

        if inicio > fin:
             print(f"   Advertencia: Rango de pagina invalido ({inicio+1} a {fin+1}). Omitiendo.")
             continue
        
        try:
            nuevo_doc = fitz.open()
            nuevo_doc.insert_pdf(doc_original, from_page=inicio, to_page=fin) 
            
            nombre_salida = os.path.join(
                carpeta_salida, 
                f"{clasificacion}_{indice}_pag_{inicio+1}_a_{fin+1}.pdf"
            )
            nuevo_doc.save(nombre_salida, garbage=4, deflate=True) 
            nuevo_doc.close()
            print(f"   Guardado: {nombre_salida} (Clasificación: {clasificacion}, Paginas {inicio+1} a {fin+1})")
            
        except Exception as e:
            print(f"   ERROR al guardar documento {clasificacion} {indice} (Paginas {inicio+1} a {fin+1}). La página podría estar corrupta. Error: {e}")
            
    doc_original.close()
    print("\nProceso de segmentación y clasificación completado!")

def obtener_modo_extraccion() -> str:
    """
    Pregunta al usuario que modo de extraccion desea usar.
    """
    while True:
        print("\nSeleccione el modo de extraccion de texto para el analisis:")
        print("  1: Hibrido (PyMuPDF nativo, luego docTR si falla - Recomendado)")
        print("  2: PyMuPDF (Solo extraccion nativa)")
        print("  3: docTR (Solo OCR, forzando la conversion a imagen)")
        
        modo = input("Ingrese 1, 2 o 3: ").strip()
        if modo in ["1", "2", "3"]:
            if modo == "1":
                return "HIBRIDO"
            elif modo == "2":
                return "NATIVO"
            else:
                return "OCR"
        else:
            print("Entrada invalida. Intente de nuevo.")


if __name__ == "__main__":
    
    patrones_demo = list(PATRONES_INICIO.keys())
    
    if not os.path.exists(PDF_ENTRADA):
        print(f"\nARCHIVO NO ENCONTRADO: '{PDF_ENTRADA}'")
        print("Creando un PDF de demostracion con patrones múltiples...")
        
        doc_temp = fitz.open()
        
        # Doc 1: Patrón de la primera clasificación (e.g., FACTURA_VENTA)
        page1 = doc_temp.new_page()
        page1.insert_text((50, 50), f"{PATRONES_INICIO[patrones_demo[0]][0]} 1", fontsize=12) 
        
        # Página de continuación
        page2 = doc_temp.new_page()
        page2.insert_text((50, 50), "Continuacion del Documento 1", fontsize=12)
        
        # Doc 2: Patrón de la segunda clasificación (e.g., LISTA_EMBALAJE)
        page3 = doc_temp.new_page()
        page3.insert_text((50, 50), f"{PATRONES_INICIO[patrones_demo[1]][0]} ABC", fontsize=12) 
        
        # Doc 3: Patrón de la última clasificación (e.g., FACTURA_AGENTE)
        page4 = doc_temp.new_page()
        page4.insert_text((50, 50), f"{PATRONES_INICIO[patrones_demo[-1]][0]} XYZ", fontsize=12) 

        doc_temp.save(PDF_ENTRADA)
        doc_temp.close()
        print(f"  -> Se creo '{PDF_ENTRADA}' con 4 páginas para la prueba usando los patrones: {PATRONES_INICIO[patrones_demo[0]][0]}, {PATRONES_INICIO[patrones_demo[1]][0]}, {PATRONES_INICIO[patrones_demo[-1]][0]}")
        print("  -> Vuelve a ejecutar el script para procesar este archivo.")
        
    else:
        modo_seleccionado = obtener_modo_extraccion()
        print(f"\nModo de extraccion seleccionado: {modo_seleccionado}")
        
        dividir_pdf_por_contenido(
            PDF_ENTRADA, 
            CARPETA_SALIDA, 
            modo_extraccion=modo_seleccionado
        )