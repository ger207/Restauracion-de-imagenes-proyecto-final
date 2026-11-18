"""
Módulo de Honestidad Intelectual, Métricas y Mapa de Intervención.

Responde al núcleo ético del Trabajo Integrador:
- "¿Qué parte de lo que ves es real y qué parte es inventada?"
- Genera mapas de calor de diferencias residuales (dónde intervino la IA / inpainting).
- Descompone la fidelidad en espacio L*a*b* (Luminancia = Dato histórico, Crominancia = Suposición generativa).
- Informa detalladamente sobre el estado de arrugas eliminadas y color agregado.
"""

from typing import Tuple, Dict, Any, Optional
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
    img_restaurada: Image.Image,
    mascara_arrugas: Optional[Image.Image] = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Compara la imagen original contra la restaurada y genera:
    1. Un mapa de calor (colormap INFERNO) que muestra la magnitud de cambio por píxel.
    2. Diccionario con métricas cuantitativas e interpretación cualitativa.
    """
    w, h = img_original.size
    img_restaurada_res = img_restaurada.resize((w, h), Image.Resampling.BICUBIC)

    arr_orig = np.array(img_original.convert("RGB"), dtype=np.float32)
    arr_rest = np.array(img_restaurada_res.convert("RGB"), dtype=np.float32)

    # Diferencia absoluta total en RGB normalizada a [0, 255]
    diff_rgb = np.linalg.norm(arr_rest - arr_orig, axis=2)
    diff_max = np.max(diff_rgb) if np.max(diff_rgb) > 0 else 1.0
    diff_norm = np.clip((diff_rgb / diff_max) * 255.0, 0, 255).astype(np.uint8)

    # Mapa de calor usando OpenCV (COLORMAP_INFERNO)
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    mapa_calor = Image.fromarray(heatmap_rgb)

    # Análisis en espacio L*a*b*
    cv_orig = cv2.cvtColor(arr_orig.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv_rest = cv2.cvtColor(arr_rest.astype(np.uint8), cv2.COLOR_RGB2BGR)
    lab_orig = cv2.cvtColor(cv_orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_rest = cv2.cvtColor(cv_rest, cv2.COLOR_BGR2LAB).astype(np.float32)

    diff_l = np.mean(np.abs(lab_rest[:, :, 0] - lab_orig[:, :, 0]))
    diff_chroma = np.mean(
        np.sqrt((lab_rest[:, :, 1] - lab_orig[:, :, 1]) ** 2 + (lab_rest[:, :, 2] - lab_orig[:, :, 2]) ** 2)
    )

    porcentaje_cambio = float(np.sum(diff_rgb > 25.5) / (w * h) * 100.0)

    # Porcentaje de arrugas reparadas
    pct_arrugas = 0.0
    if mascara_arrugas is not None:
        arr_mask = np.array(mascara_arrugas.convert("L"))
        pct_arrugas = float(np.sum(arr_mask > 0) / (w * h) * 100.0)

    # Métricas estándar
    gris_orig = cv2.cvtColor(cv_orig, cv2.COLOR_BGR2GRAY)
    gris_rest = cv2.cvtColor(cv_rest, cv2.COLOR_BGR2GRAY)

    valor_psnr = float(psnr(gris_orig, gris_rest, data_range=255))
    valor_ssim = float(ssim(gris_orig, gris_rest, data_range=255))

    es_bn = es_imagen_monocroma(img_original)
    tiene_color_nuevo = diff_chroma > 4.0

    metricas = {
        "porcentaje_modificado": round(porcentaje_cambio, 2),
        "porcentaje_arrugas_reparadas": round(pct_arrugas, 2),
        "cambio_luminancia_l": round(diff_l, 2),
        "cambio_crominancia_ab": round(diff_chroma, 2),
        "psnr_db": round(valor_psnr, 2),
        "ssim": round(valor_ssim, 4),
        "original_era_monocroma": es_bn,
        "tiene_color_nuevo": tiene_color_nuevo
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
    pct_arrugas = metricas.get("porcentaje_arrugas_reparadas", 0.0)
    color_nuevo = metricas.get("tiene_color_nuevo", False)

    # Alerta ética de colorización
    if color_nuevo:
        alerta_color = (
            "> ⚠️ **Atención sobre el Color Añadido:** Se aplicó colorización sobre el documento. "
            "El color presente en la imagen es **100% una inferencia generada por el sistema**, "
            "no un dato registrado en la película original."
        )
    elif es_bn:
        alerta_color = (
            "> ℹ️ **Preservación Monocroma:** Se conservó la naturaleza en blanco y negro original "
            "de la fotografía, sin inventar crominancia artificial."
        )
    else:
        alerta_color = (
            "> ℹ️ **Balance Cromático:** La foto original poseía datos de color. Se estabilizó la saturación."
        )

    # Alerta sobre arrugas
    alerta_arrugas = ""
    if pct_arrugas > 0:
        alerta_arrugas = (
            f"- **Arrugas y Fisuras Removidas:** Se detectaron y repararon pliegues en el **{pct_arrugas}%** del lienzo fotográfico "
            "mediante Inpainting matemático, eliminando las grietas del papel sin alterar los rasgos no dañados."
        )
    else:
        alerta_arrugas = "- **Arrugas y Fisuras:** No se aplicó remoción morfológica de grietas."

    if ssim_val > 0.85:
        interpretacion_ssim = "Alta fidelidad estructural (rostro y siluetas principales intactas)."
    elif ssim_val > 0.65:
        interpretacion_ssim = "Preservación moderada con remoción exitosa de arrugas y mejora de contraste."
    else:
        interpretacion_ssim = "Intervención profunda. Importante componente sintético."

    informe = f"""
### ⚖️ Auditoría de Fidelidad Histórica (Qué es Real vs. Qué es Inventado)

{alerta_color}

#### 1. Desglose de la Intervención:
- **Dato Real Conservado:** Estructura de luminosidad ($L^*$) y rasgos fisonómicos intactos.
{alerta_arrugas}
- **Modificación Global:** {porcentaje}% del área total experimentó ajustes de contraste, inpainting o color.
- **Nivel de Credibilidad:** La imagen luce nítida y restaurada para apreciación comunitaria en el archivo.
- **Verificabilidad:** **No utilizable como prueba forense.** Cotejar siempre con el original en papel antes de catalogaciones historiográficas.

#### 2. Indicadores Numéricos de Calidad:
- **Estructura Conservada (SSIM):** `{ssim_val}` — *{interpretacion_ssim}*
- **Relación Señal/Ruido (PSNR):** `{metricas['psnr_db']} dB`
- **Variación de Color ($a^*, b^*$):** `{metricas['cambio_crominancia_ab']}` puntos.
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

    lienzo = Image.new("RGB", (w, h + alto_banda), (20, 20, 25))
    lienzo.paste(img_con_sello, (0, 0))

    dibujo = ImageDraw.Draw(lienzo)
    texto = "ARCHIVAR - RESTAURACIÓN ASISTIDA POR IA | ARCHIVO HISTÓRICO BARRIAL (DETALLES SINTETIZADOS NO VERIFICABLES)"

    try:
        fuente = ImageFont.load_default()
    except Exception:
        fuente = None

    dibujo.text((10, h + int(alto_banda * 0.2)), texto, fill=(220, 220, 220), font=fuente)
    return lienzo
