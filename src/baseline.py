"""
Módulo de Procesamiento Digital de Imágenes Clásico (Línea Base y Detección Estructural).

Provee métodos deterministas para restaurar fotos antiguas:
1. Detección y remoción precisa de arrugas, fisuras y pliegues de papel mediante
   morfología matemática multiescala (Top-Hat), filtrado de elongación geométrica e Inpainting Telea.
   - Protege estrictamente los ojos y rasgos faciales (rechaza brillos y puntos redondos).
   - Ignora márgenes de escaneo de bordes rotos.
   - Sin costuras duras ni artefactos de dilatación excesiva.
2. Mejora de contraste adaptativo mediante CLAHE en espacio L*a*b*.
3. Reducción de ruido y preservación de bordes mediante Filtro Bilateral leve.
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
    clip_limit: float = 1.6,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Image.Image:
    """
    Aplica CLAHE suave sobre el canal L* en el espacio L*a*b*.
    El límite de recorte suave (1.6) evita quemar luces o resaltar imperfecciones de fondo.
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
    sensibilidad: int = 45
) -> np.ndarray:
    """
    Detecta fisuras y pliegues físicos del papel de forma quirúrgica.
    
    Salvaguardas críticas:
    1. Enfocado en Top-Hat (fracturas blancas donde la emulsión se abrió mostrando papel).
       NO utiliza Black-Hat sobre la cara para impedir que las pupilas, pestañas y fosas nasales sean borradas.
    2. Margen perimetral: ignora los bordes rotos blancos del escaneo para no mezclarlos con la foto.
    3. Filtrado de Elongación Geométrica: las grietas son trazos alargados (elongación >= 1.8).
       Cualquier mancha circular (como los brillos de los ojos o reflejos de córnea) es descartada.
    """
    gris = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # 1. Filtro morfológico multiescala para capturar fisuras finas y medianas
    tophat_acc = np.zeros_like(gris)
    for k in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat_acc = cv2.max(tophat_acc, cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel))

    # 2. Protección de márgenes de escaneo (evita que los marcos de papel roto sangren hacia adentro)
    borde_seguro = max(4, int(min(h, w) * 0.015))
    tophat_acc[:borde_seguro, :] = 0
    tophat_acc[-borde_seguro:, :] = 0
    tophat_acc[:, :borde_seguro] = 0
    tophat_acc[:, -borde_seguro:] = 0

    # 3. Umbral controlado por la sensibilidad del usuario [15..65]
    # Sensibilidad 45 -> umbral ~32
    umbral = int(48 - (min(max(sensibilidad, 15), 65) / 65.0) * 22)
    _, raw_mask = cv2.threshold(tophat_acc, umbral, 255, cv2.THRESH_BINARY)

    # 4. Análisis morfológico por contorno: rechazar círculos (ojos/destellos) y retener líneas (grietas)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(raw_mask)
    max_area_permitida = int(w * h * 0.025)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 6:
            continue

        if len(cnt) >= 5:
            _, (major, minor), _ = cv2.fitEllipse(cnt)
            elongacion = major / max(1.0, minor)
        else:
            _, _, bw, bh = cv2.boundingRect(cnt)
            elongacion = max(bw, bh) / max(1, min(bw, bh))

        # Una grieta es alargada (elongación >= 1.8). Los brillos de los ojos tienen elongación ~ 1.0.
        if elongacion >= 1.8 and area <= max_area_permitida:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # 5. Dilatación quirúrgica mínima (1 píxel) para sellar la hendidura sin expandir la zona
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel_dil, iterations=1)

    return clean_mask


def remover_arrugas_e_inpainting(
    pil_img: Image.Image,
    mascara_pil: Optional[Image.Image] = None,
    sensibilidad: int = 45,
    metodo: str = "telea",
    radio_inpaint: int = 3
) -> Tuple[Image.Image, Image.Image]:
    """
    Elimina las grietas del papel mediante Inpainting determinista Telea (Fast Marching Method).
    No aplica mezclas alfa posteriores para evitar generar costuras o estirar las grietas.
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
    sigma_color: float = 25.0,
    sigma_espacio: float = 25.0
) -> Image.Image:
    """
    Suavizado bilateral conservador para atenuar grano de película sin desdibujar bordes ni ojos.
    """
    cv_img = pil_to_cv2(pil_img)
    filtrada = cv2.bilateralFilter(cv_img, d=diametro, sigmaColor=sigma_color, sigmaSpace=sigma_espacio)
    return cv2_to_pil(filtrada)


def pipeline_restauracion_clasica(
    pil_img: Image.Image,
    eliminar_arrugas: bool = True,
    sensibilidad_arrugas: int = 45,
    metodo_inpaint: str = "telea",
    aplicar_clahe: bool = True,
    aplicar_bilateral: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """
    Aplica el pipeline clásico completo de la materia:
    1. Inpainting quirúrgico de arrugas respetando los ojos y bordes de escaneo.
    2. Suavizado bilateral leve.
    3. Realce de contraste adaptativo CLAHE suave.
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
