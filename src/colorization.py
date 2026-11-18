"""
Módulo de Colorización Histórica Asistida (Deep Learning en Espacio L*a*b* con Protección de Tono).

Implementa:
1. Colorización neuronal semántica (Richard Zhang et al., ECCV 2016 / OpenCV ONNX).
   - Filtro Anti-Verde de Estudio: Elimina la tendencia de redes preentrenadas a
     confundir fondos fotográficos texturados con vegetación o follaje.
   - Piel natural cálida: preserva el vector fisiológico de la piel humana (R > G > B).
   - Neutralización de sombras: el cabello y la ropa oscura conservan pigmentos
     negros/castaños naturales en lugar de adquirir tonalidades verdosas.
2. Tonalidad Sepia de época para archivos fotográficos clásicos.
3. Preservación Monocroma fiel (blanco y negro puro sin invención de color).
4. Control continuo de saturación cromática.
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
    - 'realista': Red Neuronal Profunda Zhang con calibración para retratos de archivo.
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
            # Entrada normalizada en espacio Lab
            img_rgb_norm = (cv_bgr[:, :, ::-1] * 1.0 / 255.0).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb_norm, cv2.COLOR_RGB2Lab)

            img_l = img_lab[:, :, 0]
            img_l_rs = cv2.resize(img_l, (224, 224)) - 50.0

            net.setInput(cv2.dnn.blobFromImage(img_l_rs))
            ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab_dec_us = cv2.resize(ab_dec, (w, h))

            a_chan = ab_dec_us[:, :, 0]
            b_chan = ab_dec_us[:, :, 1]

            # 1. Filtro Anti-Verde de Estudio:
            # En retratos de estudio, los valores a* < 0 generan el temido "tono verde musgo".
            # Clampeamos a* a valores >= 0.5 (ligeramente neutro/cálido).
            a_chan = np.maximum(a_chan, 0.5)

            # 2. Modulador Tonal Fisiológico:
            # En sombras profundas (pelo, pupila, ropa oscura L < 30) y en luces quemadas (L > 90),
            # la saturación debe ser mínima para evitar teñir el fondo de verde o azul.
            l_norm = img_l / 100.0
            mascara_saturacion = np.clip(np.sin(l_norm * np.pi) * 1.15, 0.25, 1.0)

            # 3. Aplicar saturación elegida por Elena
            a_chan = a_chan * mascara_saturacion * float(saturacion)
            b_chan = b_chan * mascara_saturacion * float(saturacion)

            # Reensamblado en Lab
            ab_final = np.stack((a_chan, b_chan), axis=2)
            img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_final), axis=2)
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            rgb_final = cv2.cvtColor((img_bgr_out * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_final)

    # 2. Modo Sepia de Época
    scaled = cv_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    a_channel = np.full((h, w), 5.0 * saturacion, dtype=np.float32)
    b_channel = np.full((h, w), 22.0 * saturacion, dtype=np.float32) * (l_channel > 10.0).astype(np.float32)

    lab_out = np.concatenate(
        (l_channel[:, :, np.newaxis], a_channel[:, :, np.newaxis], b_channel[:, :, np.newaxis]),
        axis=2
    )
    bgr_out = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR), 0, 1)
    rgb_final = cv2.cvtColor((bgr_out * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_final)
