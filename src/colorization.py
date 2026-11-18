"""
Módulo de Colorización Histórica Asistida (Deep Learning en Espacio L*a*b*).

Implementa:
1. Colorización neuronal semántica real utilizando el modelo ONNX oficial de OpenCV
   (Richard Zhang et al., Colorful Image Colorization, ECCV 2016).
   Predice la distribución de crominancia (a*, b*) a partir del canal de luminancia (L*),
   otorgando tonos de piel reales, pigmentos textiles naturales y fondos coherentes.
2. Tonalidad Sepia de época para archivos fotográficos clásicos.
3. Preservación Monocroma fiel (blanco y negro puro sin invención de color).
4. Control continuo de saturación mediante mezcla de canales cromáticos.
"""

import os
from typing import Optional
import numpy as np
import cv2
from PIL import Image

_NET_COLORIZATION = None
_MODELO_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "colorization",
    "submissions",
    "colorization",
    "colorization_deploy_v2_2026april.onnx"
)


def cargar_red_colorizacion() -> Optional[cv2.dnn.Net]:
    """Carga y cachea la red neuronal ONNX de colorización de OpenCV."""
    global _NET_COLORIZATION
    if _NET_COLORIZATION is not None:
        return _NET_COLORIZATION

    if os.path.exists(_MODELO_ONNX_PATH):
        try:
            print(f"[Colorización] Cargando modelo ONNX desde '{_MODELO_ONNX_PATH}'...")
            net = cv2.dnn.readNet(_MODELO_ONNX_PATH)
            _NET_COLORIZATION = net
            return _NET_COLORIZATION
        except Exception as e:
            print(f"[Colorización Error] No se pudo cargar el modelo ONNX: {e}")
            return None
    return None


def colorizar_foto_lab(
    img_pil: Image.Image,
    estilo: str = "realista",
    saturacion: float = 0.85
) -> Image.Image:
    """
    Aplica colorización sobre la fotografía:
    - 'realista': Red Neuronal Profunda Zhang (ECCV 2016 / OpenCV ONNX).
    - 'sepia': Tonalidad cálida fotográfica vintage.
    - 'monocromo': Blanco y negro puro (sin color).
    """
    if estilo == "monocromo" or saturacion <= 0.01:
        gris = img_pil.convert("L")
        return Image.merge("RGB", (gris, gris, gris))

    cv_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = cv_bgr.shape[:2]

    if estilo == "realista":
        net = cargar_red_colorizacion()
        if net is not None:
            # 1. Pipeline de Red Neuronal Profunda Zhang
            img_rgb_norm = (cv_bgr[:, :, ::-1] * 1.0 / 255.0).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb_norm, cv2.COLOR_RGB2Lab)

            img_l = img_lab[:, :, 0]
            img_l_rs = cv2.resize(img_l, (224, 224))
            img_l_rs -= 50.0  # Centrado en cero según diseño de la red

            net.setInput(cv2.dnn.blobFromImage(img_l_rs))
            ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

            # Redimensionar crominancia estimada al tamaño original de la foto
            ab_dec_us = cv2.resize(ab_dec, (w, h))

            # Ajuste de saturación multiplicando canales ab
            if saturacion != 1.0:
                ab_dec_us = ab_dec_us * float(saturacion)

            img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)
            rgb_final = cv2.cvtColor((img_bgr_out * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_final)

    # 2. Modo Sepia Clásico o Fallback
    scaled = cv_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Canales en escala Lab (a*, b* centrados en 0)
    a_channel = np.zeros((h, w), dtype=np.float32)
    b_channel = np.zeros((h, w), dtype=np.float32)

    # Sepia fotográfico: A ligeramente rojizo, B dorado cálido
    a_channel += 6.0 * saturacion
    b_channel += 24.0 * saturacion * (l_channel > 10.0).astype(np.float32)

    lab_out = np.concatenate(
        (l_channel[:, :, np.newaxis], a_channel[:, :, np.newaxis], b_channel[:, :, np.newaxis]),
        axis=2
    )
    bgr_out = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR), 0, 1)
    rgb_final = cv2.cvtColor((bgr_out * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_final)
