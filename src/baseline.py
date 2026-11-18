"""
Módulo de Procesamiento Digital de Imágenes Clásico (Línea Base).

Provee métodos deterministas tradicionales para comparar contra modelos generativos:
1. Mejora de contraste adaptativo mediante CLAHE en espacio L*a*b*.
2. Detección automática y restauración de rayones con Inpainting clásico (Telea / Navier-Stokes).
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
    Ideal para recuperar detalles en fotos históricas desvanecidas o subexpuestas.
    """
    cv_img = pil_to_cv2(pil_img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_resultado = cv2.merge((l_clahe, a_channel, b_channel))
    bgr_resultado = cv2.cvtColor(lab_resultado, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(bgr_resultado)


def detectar_rayones_morfologia(
    cv_img: np.ndarray,
    kernel_size: int = 5,
    umbral: int = 35
) -> np.ndarray:
    """
    Detecta rayones finos y grietas utilizando transformaciones morfológicas (Black-Hat).
    Devuelve una máscara binaria de 8 bits donde 255 indica daño detectado.
    """
    gris = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Kernel en cruz/elipse para capturar trazos de rayones en varias direcciones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Black Hat: Resalta elementos oscuros sobre fondo claro (grietas típicas)
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)
    
    # Top Hat: Resalta elementos claros sobre fondo oscuro (rayones reflectantes)
    tophat = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel)
    
    # Unión de candidatos a daño
    danio = cv2.add(blackhat, tophat)
    
    # Umbralizado para obtener máscara binaria limpia
    _, mascara = cv2.threshold(danio, umbral, 255, cv2.THRESH_BINARY)
    
    # Dilatación leve para cubrir bien los bordes del rayón
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mascara = cv2.dilate(mascara, kernel_dilatacion, iterations=1)
    
    return mascara


def inpainting_clasico(
    pil_img: Image.Image,
    mascara_pil: Optional[Image.Image] = None,
    metodo: str = "telea",
    radio_inpaint: int = 3
) -> Tuple[Image.Image, Image.Image]:
    """
    Restaura zonas dañadas utilizando algoritmos clásicos de inpainting de OpenCV.
    - Metodo 'telea': Algoritmo basado en método de marcha rápida (Fast Marching Method, Alexandru Telea 2004).
    - Metodo 'ns': Algoritmo basado en Navier-Stokes y fluidodinámica (Bertalmio et al. 2001).
    
    Retorna la imagen restaurada y la máscara de daño utilizada.
    """
    cv_img = pil_to_cv2(pil_img)

    if mascara_pil is not None:
        mascara = np.array(mascara_pil.convert("L"))
        # Binarizar por si la máscara tiene valores intermedios
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    else:
        # Detección morfológica automática si el usuario no aportó máscara manual
        mascara = detectar_rayones_morfologia(cv_img)

    flags = cv2.INPAINT_TELEA if metodo.lower() == "telea" else cv2.INPAINT_NS
    restaurada_cv = cv2.inpaint(cv_img, mascara, radio_inpaint, flags)

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
    aplicar_clahe: bool = True,
    aplicar_inpaint: bool = True,
    aplicar_bilateral: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """
    Aplica el pipeline clásico completo de la materia:
    1. Limpieza de ruido bilateral
    2. Inpainting morfológico de rayones
    3. Realce de contraste adaptativo CLAHE
    """
    resultado = pil_img.copy()
    mascara_usada = Image.new("L", pil_img.size, 0)

    if aplicar_bilateral:
        resultado = filtro_bilateral_limpieza(resultado)

    if aplicar_inpaint:
        resultado, mascara_usada = inpainting_clasico(resultado)

    if aplicar_clahe:
        resultado = ecualizar_clahe(resultado)

    return resultado, mascara_usada
