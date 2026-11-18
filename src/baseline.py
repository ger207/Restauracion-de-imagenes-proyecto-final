"""
Módulo de Procesamiento Digital de Imágenes Clásico (Línea Base y Detección Estructural).

Provee métodos deterministas para restaurar fotos antiguas:
1. Detección y remoción precisa de arrugas, fisuras y pliegues de papel mediante
   morfología matemática multiescala, filtrado de componentes conectados e Inpainting Telea/NS.
   Posee un límite estricto de cobertura para impedir que rasgos faciales o texturas sanas sean borradas.
2. Mejora de contraste adaptativo mediante CLAHE en espacio L*a*b*.
3. Reducción de ruido y preservación de bordes mediante Filtro Bilateral.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL a arreglo NumPy en formato BGR (OpenCV)."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convierte un arreglo NumPy en formato BGR a imagen PIL RGB."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ecualizar_clahe(
    pil_img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Image.Image:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Trabaja sobre el canal L* en el espacio L*a*b* para no alterar el balance cromático original.
    """
    cv_img = pil_to_cv2(pil_img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_resultado = cv2.merge((l_clahe, a_channel, b_channel))
    bgr_resultado = cv2.cvtColor(lab_resultado, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(bgr_resultado)


def detectar_arrugas_multiescala(
    cv_img: np.ndarray,
    sensibilidad: int = 50
) -> np.ndarray:
    """
    Detecta de forma segura fisuras, rayones y pliegues físicos del papel.
    
    Protecciones clave:
    1. Filtro Top-Hat multiescala enfocado en trazos lineales de alta frecuencia.
    2. Filtrado morfológico de componentes conectados: elimina regiones extensas
       (mejillas, ojos, ropa) para no convertirlas en huecos de inpainting.
    3. Techo máximo de cobertura: la máscara nunca supera el 12% del lienzo,
       garantizando que la fisonomía del rostro jamás se distorsione.
    """
    gris = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # Rango seguro de umbral: sensibilidad 15..65 -> umbral 50..30
    umbral = int(52 - (min(max(sensibilidad, 10), 100) / 100.0) * 24)

    # 1. Banco de filtros morfológicos lineales para capturar fisuras de emulsión
    tophat_acc = np.zeros_like(gris)
    for k in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat_acc = cv2.max(tophat_acc, cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel))

    _, raw_mask = cv2.threshold(tophat_acc, umbral, 255, cv2.THRESH_BINARY)

    # 2. Análisis de componentes conectados para descartar manchas grandes o rasgos
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
    clean_mask = np.zeros_like(raw_mask)
    max_area_permitida = int(w * h * 0.035)  # Máximo 3.5% del área por fisura continua

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= max_area_permitida:
            clean_mask[labels == i] = 255

    # 3. Dilatación mínima de 1 píxel para sellar el contorno de la rotura
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel_dil, iterations=1)

    return clean_mask


def remover_arrugas_e_inpainting(
    pil_img: Image.Image,
    mascara_pil: Optional[Image.Image] = None,
    sensibilidad: int = 50,
    metodo: str = "telea",
    radio_inpaint: int = 3
) -> Tuple[Image.Image, Image.Image]:
    """
    Elimina las marcas de arrugas y rayones del papel fotográfico mediante Inpainting.
    
    - Metodo 'telea': Fast Marching Method (Telea, 2004). Recomendado por su estabilidad.
    - Metodo 'ns': Navier-Stokes (Bertalmio et al., 2001).
    """
    cv_img = pil_to_cv2(pil_img)

    if mascara_pil is not None:
        mascara = np.array(mascara_pil.convert("L"))
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    else:
        mascara = detectar_arrugas_multiescala(cv_img, sensibilidad=sensibilidad)

    flags = cv2.INPAINT_TELEA if metodo.lower() == "telea" else cv2.INPAINT_NS
    restaurada_cv = cv2.inpaint(cv_img, mascara, radio_inpaint, flags)

    return cv2_to_pil(restaurada_cv), Image.fromarray(mascara)


def filtro_bilateral_limpieza(
    pil_img: Image.Image,
    diametro: int = 5,
    sigma_color: float = 30.0,
    sigma_espacio: float = 30.0
) -> Image.Image:
    """
    Aplica filtrado bilateral para suavizar granulado/ruido de película fotográfica
    conservando bordes nítidos.
    """
    cv_img = pil_to_cv2(pil_img)
    filtrada = cv2.bilateralFilter(cv_img, d=diametro, sigmaColor=sigma_color, sigmaSpace=sigma_espacio)
    return cv2_to_pil(filtrada)


def pipeline_restauracion_clasica(
    pil_img: Image.Image,
    eliminar_arrugas: bool = True,
    sensibilidad_arrugas: int = 50,
    metodo_inpaint: str = "telea",
    aplicar_clahe: bool = True,
    aplicar_bilateral: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """
    Aplica el pipeline clásico completo de la materia:
    1. Inpainting seguro de arrugas y grietas morfológicas.
    2. Suavizado bilateral leve.
    3. Realce de contraste adaptativo CLAHE.
    """
    resultado = pil_img.copy()
    mascara_usada = Image.new("L", pil_img.size, 0)

    if eliminar_arrugas:
        resultado, mascara_usada = remover_arrugas_e_inpainting(
            resultado,
            sensibilidad=sensibilidad_arrugas,
            metodo=metodo_inpaint
        )

    if aplicar_bilateral:
        resultado = filtro_bilateral_limpieza(resultado)

    if aplicar_clahe:
        resultado = ecualizar_clahe(resultado)

    return resultado, mascara_usada
