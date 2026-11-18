"""
Módulo de Honestidad Intelectual, Métricas y Mapa de Intervención.

Responde al núcleo ético del Trabajo Integrador:
- "¿Qué parte de lo que ves es real y qué parte es inventada?"
- Genera mapas de calor de diferencias residuales (dónde intervino la IA).
- Descompone la fidelidad en espacio L*a*b* (Luminancia = Dato histórico, Crominancia = Suposición generativa).
- Calcula métricas (PSNR, SSIM) y las traduce a explicaciones transparentes para archivistas.
- Aplica sellos editoriales de procedencia en las descargas.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def es_imagen_monocroma(img: Image.Image, tolerancia: float = 3.0) -> bool:
    """
    Determina si la imagen original es en blanco y negro / escala de grises.
    Si los canales R, G y B son prácticamente iguales, la imagen no contenía datos de color.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
    diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
    return (float(diff_rg.mean()) < tolerancia) and (float(diff_gb.mean()) < tolerancia)


def calcular_mapa_intervencion(
    img_original: Image.Image,
    img_restaurada: Image.Image
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Compara la imagen original contra la restaurada y genera:
    1. Un mapa de calor (colormap JET/INFERNO) que muestra la magnitud de cambio por píxel.
    2. Diccionario con métricas cuantitativas e interpretación cualitativa.
    """
    # Asegurar coincidencia de dimensiones para comparación píxel a píxel
    w, h = img_original.size
    img_restaurada_res = img_restaurada.resize((w, h), Image.Resampling.BICUBIC)

    arr_orig = np.array(img_original.convert("RGB"), dtype=np.float32)
    arr_rest = np.array(img_restaurada_res.convert("RGB"), dtype=np.float32)

    # 1. Diferencia absoluta total en RGB normalizada a [0, 255]
    diff_rgb = np.linalg.norm(arr_rest - arr_orig, axis=2)
    diff_max = np.max(diff_rgb) if np.max(diff_rgb) > 0 else 1.0
    diff_norm = np.clip((diff_rgb / diff_max) * 255.0, 0, 255).astype(np.uint8)

    # Mapa de calor usando OpenCV (COLORMAP_INFERNO o COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    mapa_calor = Image.fromarray(heatmap_rgb)

    # 2. Análisis en espacio L*a*b*
    cv_orig = cv2.cvtColor(arr_orig.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv_rest = cv2.cvtColor(arr_rest.astype(np.uint8), cv2.COLOR_RGB2BGR)
    lab_orig = cv2.cvtColor(cv_orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_rest = cv2.cvtColor(cv_rest, cv2.COLOR_BGR2LAB).astype(np.float32)

    diff_l = np.mean(np.abs(lab_rest[:, :, 0] - lab_orig[:, :, 0]))
    diff_chroma = np.mean(
        np.sqrt((lab_rest[:, :, 1] - lab_orig[:, :, 1]) ** 2 + (lab_rest[:, :, 2] - lab_orig[:, :, 2]) ** 2)
    )

    # Porcentaje de píxeles modificados significativamente (umbral de cambio > 10%)
    porcentaje_cambio = float(np.sum(diff_rgb > 25.5) / (w * h) * 100.0)

    # Métricas estándar
    gris_orig = cv2.cvtColor(cv_orig, cv2.COLOR_BGR2GRAY)
    gris_rest = cv2.cvtColor(cv_rest, cv2.COLOR_BGR2GRAY)

    valor_psnr = float(psnr(gris_orig, gris_rest, data_range=255))
    valor_ssim = float(ssim(gris_orig, gris_rest, data_range=255))

    es_bn = es_imagen_monocroma(img_original)

    metricas = {
        "porcentaje_modificado": round(porcentaje_cambio, 2),
        "cambio_luminancia_l": round(diff_l, 2),
        "cambio_crominancia_ab": round(diff_chroma, 2),
        "psnr_db": round(valor_psnr, 2),
        "ssim": round(valor_ssim, 4),
        "original_era_monocroma": es_bn,
    }

    return mapa_calor, metricas


def generar_informe_honestidad_md(metricas: Dict[str, Any]) -> str:
    """
    Construye el informe visual y transparente para la persona usuaria (Elena).
    Distingue lo Real, lo Ficticio, lo Creíble y lo Verificable.
    """
    es_bn = metricas["original_era_monocroma"]
    ssim_val = metricas["ssim"]
    porcentaje = metricas["porcentaje_modificado"]

    alerta_color = ""
    if es_bn:
        alerta_color = (
            "> ⚠️ **Atención sobre el color:** La fotografía original fue identificada como monocromática. "
            "Cualquier tonalidad o color presente en la imagen restaurada es **100% una inferencia estadística generada por la IA**, "
            "no un dato histórico registrado en la película original."
        )
    else:
        alerta_color = (
            "> ℹ️ **Balance cromático:** La foto original poseía datos de color. La IA ajustó la saturación "
            "y eliminó desvanecimiento químico."
        )

    # Interpretación de SSIM para Elena
    if ssim_val > 0.85:
        interpretacion_ssim = "Alta preservación estructural (formas, siluetas y rostros originales se mantuvieron intactos)."
    elif ssim_val > 0.65:
        interpretacion_ssim = "Preservación moderada. Se han reconstruido texturas y eliminado rayones notorios."
    else:
        interpretacion_ssim = "Intervención profunda. Muchas zonas fueron sintetizadas nuevamente por el modelo."

    informe = f"""
### ⚖️ Auditoría de Fidelidad Histórica (Qué es Real vs. Qué es Inventado)

{alerta_color}

#### 1. Desglose de la Restauración:
- **Dato Real Conservado:** La estructura de luminancia ($L^*$) y contornos principales del documento histórico.
- **Detalle Ficticio (Generado):** Micro-texturas reconstruidas en zonas de daño ({porcentaje}% de la imagen intervenida significativamente).
- **Nivel de Credibilidad:** La imagen luce verosímil y estéticamente recuperada para visualización comunitaria.
- **Verificabilidad:** **No utilizable como peritaje judicial o evidencia forense.** Cotejar siempre con el original en papel antes de catalogaciones historiográficas.

#### 2. Indicadores de Fidelidad Documental:
- **Estructura Conservada (SSIM):** `{ssim_val}` — *{interpretacion_ssim}*
- **Relación Señal/Ruido (PSNR):** `{metricas['psnr_db']} dB` (mayor valor = menor distorsión respecto al original).
- **Variación de Crominancia ($a^*, b^*$):** `{metricas['cambio_crominancia_ab']}` puntos de coloración estimada.
"""
    return informe.strip()


def estampar_sello_archivistico(img: Image.Image) -> Image.Image:
    """
    Inserta una discreta banda editorial en la base de la imagen para asegurar transparencia
    al exportar o exhibir el archivo en la biblioteca barrial.
    """
    img_con_sello = img.copy()
    w, h = img_con_sello.size
    alto_banda = max(24, int(h * 0.04))

    # Crear lienzo extendido con banda inferior gris oscuro/negro
    lienzo = Image.new("RGB", (w, h + alto_banda), (20, 20, 25))
    lienzo.paste(img_con_sello, (0, 0))

    dibujo = ImageDraw.Draw(lienzo)
    texto = "ARCHIVAR - RESTAURACIÓN ASISTIDA POR IA | ARCHIVO HISTÓRICO BARRIAL (DETALLES SINTETIZADOS NO VERIFICABLES)"

    # Dibujar texto centrado en la banda
    try:
        # Intentar cargar fuente estándar de sistema si existe
        fuente = ImageFont.load_default()
    except Exception:
        fuente = None

    dibujo.text((10, h + int(alto_banda * 0.2)), texto, fill=(220, 220, 220), font=fuente)
    return lienzo
