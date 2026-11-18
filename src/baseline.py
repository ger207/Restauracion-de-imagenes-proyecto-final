"""
Módulo de Procesamiento Digital de Imágenes Clásico (Línea Base y Herramientas Estructurales).

Provee métodos deterministas tradicionales para restaurar fotos antiguas:
1. Detección y remoción profunda de arrugas, fisuras y pliegues de papel mediante
   morfología matemática multiescala (Top-Hat / Black-Hat) e Inpainting (Telea / Navier-Stokes).
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
    clip_limit: float = 2.5,
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
    sensibilidad: int = 40
) -> np.ndarray:
    """
    Detecta arrugas, pliegues pronunciados y fisuras de papel fotográfico
    mediante un banco de filtros morfológicos multiescala.
    
    - Top-Hat: Detecta grietas brillantes (donde la emulsión se desprendió exponiendo el papel blanco).
    - Black-Hat: Detecta grietas oscuras o sombras producidas por pliegues.
    - sensibilidad: Rango [10, 70]. Mayor sensibilidad detecta más trazos finos.
    """
    gris = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    tophat_acc = np.zeros_like(gris)
    blackhat_acc = np.zeros_like(gris)

    # Varios tamaños de kernel para capturar desde rayones finos hasta pliegues gruesos
    for k in [3, 7, 11, 15]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat_acc = cv2.max(tophat_acc, cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel))
        blackhat_acc = cv2.max(blackhat_acc, cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel))

    danio = cv2.add(tophat_acc, blackhat_acc)

    # Umbral adaptado al nivel de sensibilidad
    umbral = max(12, int(68 - sensibilidad))
    _, mascara = cv2.threshold(danio, umbral, 255, cv2.THRESH_BINARY)

    # Dilatación controlada para asegurar que el inpainting cubra los bordes ásperos
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.dilate(mascara, kernel_dilatacion, iterations=1)

    return mascara


def remover_arrugas_e_inpainting(
    pil_img: Image.Image,
    mascara_pil: Optional[Image.Image] = None,
    sensibilidad: int = 40,
    metodo: str = "telea",
    radio_inpaint: int = 5
) -> Tuple[Image.Image, Image.Image]:
    """
    Elimina arrugas y rayones del papel fotográfico mediante Inpainting.
    
    - Metodo 'telea': Fast Marching Method (Telea, 2004).
    - Metodo 'ns': Navier-Stokes (Bertalmio et al., 2001).
    - Aplica un difuminado guiado bilateral en las zonas reparadas para que no queden costuras duras.
    """
    cv_img = pil_to_cv2(pil_img)

    if mascara_pil is not None:
        mascara = np.array(mascara_pil.convert("L"))
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    else:
        mascara = detectar_arrugas_multiescala(cv_img, sensibilidad=sensibilidad)

    flags = cv2.INPAINT_TELEA if metodo.lower() == "telea" else cv2.INPAINT_NS
    restaurada_cv = cv2.inpaint(cv_img, mascara, radio_inpaint, flags)

    # Suavizado suave localizado en la costura de la reparación
    bilateral = cv2.bilateralFilter(restaurada_cv, d=7, sigmaColor=50, sigmaSpace=50)
    mascara_float = (mascara.astype(float) / 255.0)[:, :, np.newaxis]
    restaurada_cv = (bilateral.astype(float) * mascara_float + restaurada_cv.astype(float) * (1.0 - mascara_float)).astype(np.uint8)

    return cv2_to_pil(restaurada_cv), Image.fromarray(mascara)


def filtro_bilateral_limpieza(
    pil_img: Image.Image,
    diametro: int = 7,
    sigma_color: float = 50.0,
    sigma_espacio: float = 50.0
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
    sensibilidad_arrugas: int = 40,
    metodo_inpaint: str = "telea",
    aplicar_clahe: bool = True,
    aplicar_bilateral: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """
    Aplica el pipeline clásico completo de la materia:
    1. Eliminación profunda de arrugas y rayones con Inpainting morfológico.
    2. Suavizado bilateral de ruido de emulsión.
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
