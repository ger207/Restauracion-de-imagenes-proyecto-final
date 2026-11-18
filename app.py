"""
Aplicación Principal: Restaurador y Analizador de Fotografías Históricas
Caso de uso: Archivo Histórico Barrial (Persona Usuaria: Elena)
Materia: Procesamiento Digital de Imágenes e Introducción a Visión por Computadora (IFTS 24)

Ejecución:
    uv run app.py
    uv run app.py --share  (para generar enlace público o en Google Colab)
"""

import sys
import os
from typing import Tuple, Optional
import gradio as gr
from PIL import Image

# Importación de módulos propios
from src.baseline import pipeline_restauracion_clasica
from src.diffusion import restaurar_con_difusion
from src.honesty import (
    calcular_mapa_intervencion,
    generar_informe_honestidad_md,
    estampar_sello_archivistico
)
from src.analysis import informe_completo_vlm


def procesar_fotografia(
    imagen_input: Optional[Image.Image],
    metodo: str,
    intensidad: float,
    activar_vlm: bool
) -> Tuple[Image.Image, Image.Image, str, str, Image.Image]:
    """
    Función orquestadora principal llamada por el botón de Gradio.
    """
    if imagen_input is None:
        raise gr.Error("Por favor, subí una fotografía o seleccioná uno de los ejemplos históricos del archivo.")

    img_rgb = imagen_input.convert("RGB")
    nota_tecnica = ""

    # 1. Ejecución según el método elegido por Elena
    if "Línea Base Clásica" in metodo:
        restaurada, _ = pipeline_restauracion_clasica(img_rgb)
        nota_tecnica = "Restauración determinista aplicada con técnicas clásicas de PDI (Filtro Bilateral, Inpainting Telea y CLAHE)."
    elif "Híbrido" in metodo:
        # Preprocesamiento clásico seguido de refinamiento por difusión
        base, _ = pipeline_restauracion_clasica(img_rgb)
        restaurada, detalle_dif = restaurar_con_difusion(base, intensidad=intensidad * 0.75)
        nota_tecnica = f"Pipeline híbrido: Preprocesamiento clásico + {detalle_dif}"
    else:
        # Difusión directa
        restaurada, detalle_dif = restaurar_con_difusion(img_rgb, intensidad=intensidad)
        nota_tecnica = detalle_dif

    # 2. Mapa de intervención y métricas de honestidad
    mapa_calor, metricas = calcular_mapa_intervencion(img_rgb, restaurada)
    informe_etico = generar_informe_honestidad_md(metricas)

    # 3. Análisis semántico con VLM
    if activar_vlm:
        informe_vlm = informe_completo_vlm(img_rgb, restaurada, metricas)
    else:
        informe_vlm = (
            "### 👁️ Análisis de Contenido y Semántica (VLM)\n"
            "*Análisis VLM desactivado por la usuaria para procesamiento rápido.*"
        )

    # 4. Sello de procedencia para descarga transparente
    imagen_archivistica = estampar_sello_archivistico(restaurada)

    return (
        restaurada,
        mapa_calor,
        informe_etico,
        informe_vlm + f"\n\n**Detalle del pipeline:** {nota_tecnica}",
        imagen_archivistica
    )


def construir_interfaz() -> gr.Blocks:
    """Construye la interfaz de usuario con Principios de Human-AI Interaction (HAI)."""
    
    # CSS personalizado suave y amigable para el archivo de Elena
    tema = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone"
    )

    with gr.Blocks(title="Restaurador de Archivos Históricos - Archivo Barrial", theme=tema) as demo:
        gr.Markdown(
            """
            # 🏛️ Restaurador y Analizador de Archivos Fotográficos Históricos
            ### Sistema Asistido por Inteligencia Artificial y Procesamiento Digital de Imágenes
            **Persona Usuaria:** Elena (Archivista aficionada de la Biblioteca Barrial)  
            **Materia:** Procesamiento Digital de Imágenes | IFTS 24 (2026) | Estudiante: Germán Le Piani

            ---
            > ⚖️ **Aviso de Transparencia Archivística (HAI):**  
            > *Cuando un modelo generativo procesa una fotografía antigua, la información deteriorada o ausente no se "descubre", sino que se **reconstruye de forma plausible**. Todo color y detalle de alta frecuencia generado por IA es una **suposición probabilística** y no debe utilizarse como evidencia documental incontrovertible.*
            """
        )

        with gr.Row():
            # Columna izquierda: Entrada y Parámetros
            with gr.Column(scale=1):
                gr.Markdown("### 📥 1. Fotografía de Archivo")
                entrada_img = gr.Image(
                    type="pil",
                    label="Subir foto escaneada o seleccionar ejemplo",
                    sources=["upload", "clipboard"]
                )

                metodo_selector = gr.Radio(
                    choices=[
                        "Restauración Generativa por Difusión (Hugging Face)",
                        "Línea Base Clásica (CLAHE + Inpainting OpenCV)",
                        "Pipeline Híbrido (PDI Clásico + Difusión)"
                    ],
                    value="Restauración Generativa por Difusión (Hugging Face)",
                    label="Método de Restauración",
                    info="Elegí entre IA generativa o matemática clásica de PDI para comparar resultados."
                )

                slider_intensidad = gr.Slider(
                    minimum=0.15,
                    maximum=0.75,
                    value=0.35,
                    step=0.05,
                    label="Nivel de Intervención IA (Strength)",
                    info="Bajo (0.2-0.35): Fidelidad histórica máxima. Alto (>0.5): Reconstrucción intensa (mayor invención de detalles)."
                )

                check_vlm = gr.Checkbox(
                    value=True,
                    label="Ejecutar análisis visual semántico con VLM (SmolVLM2)"
                )

                btn_procesar = gr.Button("✨ Iniciar Restauración y Auditoría", variant="primary", size="lg")

                # Ejemplos precargados
                ejemplos_dir = "assets/ejemplos"
                ejemplos_lista = []
                if os.path.exists(ejemplos_dir):
                    ejemplos_lista = [
                        [os.path.join(ejemplos_dir, f), "Restauración Generativa por Difusión (Hugging Face)", 0.35, True]
                        for f in sorted(os.listdir(ejemplos_dir)) if f.lower().endswith((".jpg", ".png", ".jpeg"))
                    ]

                if ejemplos_lista:
                    gr.Examples(
                        examples=ejemplos_lista,
                        inputs=[entrada_img, metodo_selector, slider_intensidad, check_vlm],
                        label="Fotografías históricas de muestra (Archivo Barrial)"
                    )

            # Columna derecha: Resultados y Auditoría
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ 2. Resultados Visuales y Comparación")
                
                with gr.Tabs():
                    with gr.TabItem("Vista Restaurada"):
                        salida_restaurada = gr.Image(type="pil", label="Fotografía Restaurada")

                    with gr.TabItem("Mapa de Intervención (Calor)"):
                        salida_heatmap = gr.Image(
                            type="pil",
                            label="Zonas de Mayor Intervención (Blanco/Rojo = Gran cambio; Negro/Violeta = Original intacto)"
                        )
                        gr.Markdown(
                            "*El mapa residual resalta en tonos cálidos las coordenadas exactas donde el modelo modificó la fotografía.*"
                        )

                    with gr.TabItem("Descarga con Sello de Archivo"):
                        salida_sello = gr.Image(
                            type="pil",
                            label="Fotografía con sello ético de procedencia (lista para catalogación)"
                        )

        # Fila inferior: Informes de Honestidad y Análisis Semántico
        with gr.Row():
            with gr.Column(scale=1):
                salida_informe_etico = gr.Markdown()
            with gr.Column(scale=1):
                salida_informe_vlm = gr.Markdown()

        # Vinculación de eventos
        btn_procesar.click(
            fn=procesar_fotografia,
            inputs=[entrada_img, metodo_selector, slider_intensidad, check_vlm],
            outputs=[
                salida_restaurada,
                salida_heatmap,
                salida_informe_etico,
                salida_informe_vlm,
                salida_sello
            ]
        )

    return demo


def main():
    demo = construir_interfaz()
    share_public = "--share" in sys.argv
    # Ejecución local o compartida
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=share_public
    )


if __name__ == "__main__":
    main()
