"""
Módulo de Procesamiento Digital de Imágenes Clásico (Línea Base y Detección Estructural).

Provee métodos deterministas para restaurar fotos antiguas:
1. Detección y remoción precisa de arrugas, fisuras y pliegues de papel mediante
   morfología matemática multiescala (Top-Hat), análisis de componentes conectados
   y protección facial de ojos mediante detector LBP Cascade (scikit-image).
   - Detecta fisuras delgadas y ramificadas.
   - Protege los ojos y la mirada al 100% excluyendo la zona ocular de la máscara.
   - Respeta los márgenes del escaneo para no sangrar bordes de papel roto.
2. Mejora de contraste adaptativo mediante CLAHE en espacio L*a*b*.
3. Reducción de ruido y preservación de bordes mediante Filtro Bilateral leve.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from skimage.feature import Cascade
from skimage import data

# Detector facial LBP en memoria para protección de ojos
_FACE_DETECTOR = None


def obtener_detector_rostro():
    """Carga y cachea el detector frontal LBP de scikit-image."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        try:
            _FACE_DETECTOR = Cascade(data.lbp_frontal_face_cascade_filename())
        except Exception:
            _FACE_DETECTOR = None
    return _FACE_DETECTOR


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
    Detecta arrugas, pliegues y fracturas de emulsión fotográfica blanca.
    
    Salvaguardas clave:
    1. Detector LBP para aislar el rostro y proteger los ojos:
       Cero píxeles de inpainting en córneas, párpados o pupilas.
    2. Análisis por componentes conectados de píxeles reales (sin descartar líneas finas).
    3. Protección perimetral de bordes de escaneo.
    """
    gris = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # 1. Protección de ojos con detector facial LBP
    mascara_ojos = np.zeros_like(gris)
    detector = obtener_detector_rostro()
    if detector is not None:
        try:
            faces = detector.detect_multi_scale(
                gris,
                scale_factor=1.2,
                step_ratio=1,
                min_size=(max(40, int(min(h, w) * 0.12)), max(40, int(min(h, w) * 0.12))),
                max_size=(int(min(h, w) * 0.8), int(min(h, w) * 0.8))
            )
            for f in faces:
                r, c, fw, fh = f['r'], f['c'], f['width'], f['height']
                ojo_y1 = max(0, int(r + fh * 0.25))
                ojo_y2 = min(h, int(r + fh * 0.54))
                ojo_x1 = max(0, int(c + fw * 0.12))
                ojo_x2 = min(w, int(c + fw * 0.88))
                mascara_ojos[ojo_y1:ojo_y2, ojo_x1:ojo_x2] = 255
        except Exception:
            pass

    # 2. Banco de filtros morfológicos Top-Hat para fracturas de emulsión
    tophat_acc = np.zeros_like(gris)
    for k in [3, 5, 7, 11]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tophat_acc = cv2.max(tophat_acc, cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, kernel))

    # 3. Margen perimetral para ignorar bordes rotos del escaneo
    borde_seguro = max(5, int(min(h, w) * 0.015))
    tophat_acc[:borde_seguro, :] = 0
    tophat_acc[-borde_seguro:, :] = 0
    tophat_acc[:, :borde_seguro] = 0
    tophat_acc[:, -borde_seguro:] = 0

    # 4. Umbral calibrado por sensibilidad [15..65]
    # Sensibilidad 45 -> umbral 27; Sensibilidad 65 -> umbral 20
    s_val = min(max(sensibilidad, 15), 65)
    umbral = int(38 - ((s_val - 15) / 50.0) * 19)
    _, raw_mask = cv2.threshold(tophat_acc, umbral, 255, cv2.THRESH_BINARY)

    # 5. Componentes conectados analizando píxeles reales
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
    clean_mask = np.zeros_like(raw_mask)
    max_area_permitida = int(w * h * 0.04)

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        # Retener componentes con al menos 4 píxeles (las grietas delgadas se conservan)
        if 4 <= area_px <= max_area_permitida:
            clean_mask[labels == i] = 255

    # 6. Dilatación quirúrgica de 1 píxel para unir trazos rotos
    clean_mask = cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # 7. BLINDAJE OCULAR: Asegurar que los ojos tengan 0 píxeles de inpainting
    clean_mask[mascara_ojos > 0] = 0

    return clean_mask


def remover_arrugas_e_inpainting(
    pil_img: Image.Image,
    mascara_pil: Optional[Image.Image] = None,
    sensibilidad: int = 50,
    metodo: str = "telea",
    radio_inpaint: int = 3
) -> Tuple[Image.Image, Image.Image]:
    """
    Elimina las grietas del papel mediante Inpainting determinista Telea (Fast Marching Method).
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
    sensibilidad_arrugas: int = 50,
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
