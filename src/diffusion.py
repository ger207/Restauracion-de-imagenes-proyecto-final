"""
Módulo de Restauración mediante Modelos de Difusión de Hugging Face.

Implementa:
1. Pipeline generativo Image-to-Image / Inpainting.
2. Control de intensidad ('strength'): qué tanto se permite al modelo alterar la imagen original.
3. Carga diferida (lazy loading) y cacheo de modelos para optimizar memoria RAM/VRAM.
4. Soporte automático para GPU (CUDA fp16) y CPU (fp32).
5. Modo de ejecución resiliente con mensajes de diagnóstico claros para la interfaz.
"""

from typing import Optional, Tuple
import os
import torch
from PIL import Image

# Pipeline global cacheado
_PIPELINE_DIFUSION = None
_MODELO_ACTUAL = None

# Modelo por defecto: sd-turbo para inferencia rápida (1-4 pasos) o SD 1.5 para fidelidad estándar
MODELO_DIFUSION_DEFAULT = os.getenv("DIFFUSION_MODEL_ID", "stabilityai/sd-turbo")


def obtener_dispositivo() -> Tuple[torch.device, torch.dtype]:
    """Detecta si hay acelerador CUDA disponible o si se ejecuta en CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def cargar_pipeline_difusion(model_id: str = MODELO_DIFUSION_DEFAULT):
    """
    Carga el pipeline de difusión una sola vez en memoria (patrón Singleton/Cache).
    Aplica optimizaciones según el dispositivo disponible.
    """
    global _PIPELINE_DIFUSION, _MODELO_ACTUAL

    if _PIPELINE_DIFUSION is not None and _MODELO_ACTUAL == model_id:
        return _PIPELINE_DIFUSION

    device, dtype = obtener_dispositivo()
    print(f"[Difusión] Cargando modelo '{model_id}' en dispositivo '{device}' con dtype {dtype}...")

    from diffusers import AutoPipelineForImage2Image

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True
    )

    if device.type == "cuda":
        # Habilitar optimizaciones de memoria en GPU si es necesario
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    _PIPELINE_DIFUSION = pipe
    _MODELO_ACTUAL = model_id
    return _PIPELINE_DIFUSION


def restaurar_con_difusion(
    img_original: Image.Image,
    prompt: str = "vintage historical photograph, highly detailed, restored archival quality, clear textures, realistic, sharp focus",
    prompt_negativo: str = "blurry, distorted, artifacts, scratches, modern digital noise, watermark, oversaturated",
    intensidad: float = 0.35,
    pasos: int = 4,
    guidance_scale: float = 1.5,
    model_id: str = MODELO_DIFUSION_DEFAULT
) -> Tuple[Image.Image, str]:
    """
    Ejecuta el proceso de difusión sobre la fotografía histórica.
    
    Parámetros:
    - intensidad (strength): Rango [0.0, 1.0]. 
      Valores bajos (0.2 - 0.4) conservan la estructura histórica fiel y limpian rayones.
      Valores altos (> 0.6) redibujan la escena y aumentan el componente ficticio/inventado.
    - pasos: Número de pasos de desnoising (en sd-turbo 2 a 6 pasos son óptimos).
    """
    # Asegurar tamaño múltiplo de 8 y tamaño manejable para rendimiento
    ancho, alto = img_original.size
    max_dim = 768 if torch.cuda.is_available() else 512
    escala = min(1.0, max_dim / max(ancho, alto))
    nuevo_ancho = int((ancho * escala) // 8 * 8)
    nuevo_alto = int((alto * escala) // 8 * 8)

    img_input = img_original.convert("RGB").resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    try:
        pipe = cargar_pipeline_difusion(model_id)
        device, _ = obtener_dispositivo()

        # Si el modelo es turbo, guidance_scale suele ser 0.0 o bajo
        es_turbo = "turbo" in model_id.lower()
        cfg_scale = 0.0 if es_turbo else guidance_scale
        num_steps = max(2, pasos) if es_turbo else max(15, pasos)

        with torch.inference_mode():
            resultado = pipe(
                prompt=prompt,
                negative_prompt=prompt_negativo if not es_turbo else None,
                image=img_input,
                strength=intensidad,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale,
            ).images[0]

        # Reescalar de vuelta al tamaño original para mantener dimensiones
        resultado = resultado.resize((ancho, alto), Image.Resampling.BICUBIC)
        detalles = f"Procesado exitosamente con `{model_id}` en {device.type.upper()} ({num_steps} pasos, intensidad {intensidad})."
        return resultado, detalles

    except Exception as e:
        # Modo de contingencia local: si falla la descarga o falta memoria, informa claramente
        print(f"[Aviso] No se pudo ejecutar el pipeline de difusión: {e}")
        detalles_error = (
            f"Nota de ejecución: El modelo `{model_id}` requiere mayor VRAM/tiempo en CPU. "
            f"Para aceleración por GPU ejecuta el notebook de Colab adjunto. Error: {str(e)[:120]}"
        )
        return img_input.resize((ancho, alto)), detalles_error
