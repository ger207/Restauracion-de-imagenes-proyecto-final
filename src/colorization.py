"""
Módulo de Colorización Histórica Asistida (Espacio L*a*b*).

Permite a la persona usuaria (Elena) decidir si desea otorgar color a una fotografía
monocromática o preservarla en blanco y negro original:
1. Estilo 'realista': Síntesis cromática para retratos y escenas (tonos de piel, ropa y fondos).
2. Estilo 'sepia': Tonalidad cálida fotográfica clásica de época.
3. Estilo 'monocromo': Conserva el blanco y negro original restaurado sin alteración cromática.

Concepto PDI: La luminancia (L*) se conserva de forma determinista (dato histórico),
mientras que los canales a* y b* se sintetizan según la paleta elegida (dato estimado).
"""

from typing import Tuple
import numpy as np
import cv2
from PIL import Image


def colorizar_foto_lab(
    img_pil: Image.Image,
    estilo: str = "realista",
    saturacion: float = 0.75
) -> Image.Image:
    """
    Aplica colorización paramétrica en espacio L*a*b* preservando el 100% de la luminancia original.
    
    Parámetros:
    - estilo: 'realista', 'sepia' o 'monocromo'.
    - saturacion: Nivel de vivacidad del color [0.0 a 1.0].
    """
    if estilo == "monocromo" or saturacion <= 0.01:
        # Modo blanco y negro puro
        gris = img_pil.convert("L")
        return Image.merge("RGB", (gris, gris, gris))

    arr_rgb = np.array(img_pil.convert("RGB"))
    cv_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB)
    
    l_channel = lab[:, :, 0]
    h, w = l_channel.shape
    l_norm = l_channel.astype(np.float32) / 255.0

    # Valor neutro en Lab (OpenCV mapea -128..127 a 0..255, centro = 128)
    a_channel = np.full((h, w), 128.0, dtype=np.float32)
    b_channel = np.full((h, w), 128.0, dtype=np.float32)

    if estilo == "sepia":
        # Tonalidad sepia: A* ligeramente cálido, B* amarillento dorado
        a_channel += 8.0 * saturacion
        b_channel += 26.0 * saturacion * (l_norm > 0.08).astype(np.float32)

    elif estilo == "realista":
        # Distribución cromática adaptativa para fisonomía y escenas históricas
        
        # 1. Tonalidad de piel humana (predomina en tonos medios: L entre 0.30 y 0.82)
        mascara_piel = np.exp(-((l_norm - 0.58) ** 2) / 0.075)
        a_channel += (19.0 * mascara_piel * saturacion)
        b_channel += (23.0 * mascara_piel * saturacion)

        # 2. Sombras, cabello y vestimenta oscura (L entre 0.08 y 0.35)
        # Tono ligeramente cálido / tierra natural
        mascara_sombras = ((l_norm < 0.35) & (l_norm > 0.08)).astype(np.float32)
        a_channel += (4.0 * mascara_sombras * saturacion)
        b_channel += (8.0 * mascara_sombras * saturacion)

        # 3. Luces altas y fondos claros (L > 0.78): sutil calidez ambiental
        mascara_luces = (l_norm > 0.78).astype(np.float32)
        b_channel += (10.0 * mascara_luces * saturacion)

    # Reensamblado en espacio Lab
    lab_nuevo = np.zeros_like(lab)
    lab_nuevo[:, :, 0] = l_channel
    lab_nuevo[:, :, 1] = np.clip(a_channel, 0, 255).astype(np.uint8)
    lab_nuevo[:, :, 2] = np.clip(b_channel, 0, 255).astype(np.uint8)

    bgr_salida = cv2.cvtColor(lab_nuevo, cv2.COLOR_LAB2BGR)
    rgb_salida = cv2.cvtColor(bgr_salida, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_salida)
