"""
Aplicación Principal: Restaurador y Analizador de Fotografías Históricas
Caso de uso: Archivo Histórico Barrial (Persona Usuaria: Elena)
Materia: Procesamiento Digital de Imágenes e Introducción a Visión por Computadora (IFTS 24)

Ejecución:
    uv run app.py
    uv run app.py --share  (para generar enlace público o en Google Colab)
"""

import sys
from typing import Tuple, Optional
import gradio as gr
from PIL import Image

# Módulos propios
from src.baseline import (
    remover_arrugas_e_inpainting,
    ecualizar_clahe,
    filtro_bilateral_limpieza
)
from src.diffusion import restaurar_con_difusion
from src.honesty import (
    calcular_mapa_intervencion,
    generar_informe_honestidad_md,
    estampar_sello_archivistico
)
from src.analysis import informe_completo_vlm


def procesar_fotografia(
    imagen_input: Optional[Image.Image],
    eliminar_arrugas: bool,
    sensibilidad_arrugas: int,
    metodo_inpaint: str,
    proteger_identidad: bool,
    metodo_pipeline: str,
    intensidad_ia: float,
    activar_vlm: bool
) -> Tuple[Image.Image, Image.Image, Image.Image, str, str, Image.Image]:
    """
    Función orquestadora principal centrada en restauración histórica fiel:
    1. Remoción quirúrgica de arrugas y fracturas físicas del papel.
    2. Preservación estricta de la identidad facial de la persona retratada.
    3. Realce generativo por difusión respetando la monocromía histórica.
    4. Auditoría ética transparente y análisis semántico VLM.
    """
    if imagen_input is None:
        raise gr.Error("Por favor, subí una fotografía antigua escaneada para comenzar.")

    try:
        img_rgb = imagen_input.convert("RGB")
        resultado = img_rgb.copy()
        mascara_arrugas = Image.new("L", img_rgb.size, 0)
        pasos_aplicados = []

        # 1. Remoción quirúrgica de arrugas y grietas (Inpainting con blindaje ocular)
        if eliminar_arrugas:
            resultado, mascara_arrugas = remover_arrugas_e_inpainting(
                resultado,
                sensibilidad=sensibilidad_arrugas,
                metodo=metodo_inpaint
            )
            pasos_aplicados.append(f"Inpainting de arrugas ({metodo_inpaint.upper()}, sens. {sensibilidad_arrugas})")

        # 2. Suavizado bilateral leve y CLAHE suave de luminancia
        resultado = filtro_bilateral_limpieza(resultado)
        resultado = ecualizar_clahe(resultado)
        pasos_aplicados.append("Filtrado bilateral + CLAHE de contraste")

        # 3. Realce por Difusión (si fue seleccionado)
        detalle_difusion = "Sin difusión generativa adicional."
        if "Difusión" in metodo_pipeline:
            resultado, detalle_difusion = restaurar_con_difusion(
                resultado,
                intensidad=intensidad_ia,
                proteger_identidad=proteger_identidad
            )
            pasos_aplicados.append(f"Difusión ({detalle_difusion})")

        # 4. Asegurar preservación de la monocromía histórica original
        # En retratos de archivo, la luminancia es el dato real y se evita inventar tintes artificiales
        pasos_aplicados.append("Preservación de blanco y negro documental histórico")

        # 5. Auditoría ética y mapa residual
        mapa_calor, metricas = calcular_mapa_intervencion(img_rgb, resultado, mascara_arrugas=mascara_arrugas)
        informe_etico = generar_informe_honestidad_md(metricas)

        # 6. Análisis semántico con VLM
        if activar_vlm:
            informe_vlm = informe_completo_vlm(img_rgb, resultado, metricas)
        else:
            informe_vlm = (
                "### 👁️ Análisis de Contenido y Semántica (VLM)\n"
                "*Análisis VLM omitido por configuración de usuario.*"
            )

        # 7. Sello editorial
        imagen_con_sello = estampar_sello_archivistico(resultado)

        resumen_pipeline = f"\n\n**Flujo aplicado:** " + " ➔ ".join(pasos_aplicados)

        return (
            resultado,
            mascara_arrugas,
            mapa_calor,
            informe_etico,
            informe_vlm + resumen_pipeline,
            imagen_con_sello
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Aviso durante el procesamiento: {str(e)}")


def construir_interfaz() -> gr.Blocks:
    """Construye la interfaz de usuario orientada a Elena con Gradio 6."""
    
    with gr.Blocks(title="Restaurador de Archivos Históricos - Archivo Barrial") as demo:
        gr.Markdown(
            """
            # 🏛️ Restaurador y Analizador de Archivos Fotográficos Históricos
            ### Sistema Asistido por Inteligencia Artificial y Procesamiento Digital de Imágenes
            **Persona Usuaria:** Elena (Archivista aficionada de la Biblioteca Barrial)  
            **Materia:** Procesamiento Digital de Imágenes | IFTS 24 (2026) | Estudiante: Germán Le Piani

            ---
            > ⚖️ **Aviso de Transparencia Archivística (HAI):**  
            > *La información deteriorada o ausente no se "descubre", sino que se **reconstruye de forma verosímil**. Todo detalle sintetizado por el modelo es una **inferencia probabilística** y debe verificarse antes de utilizarse con fines historiográficos.*
            """
        )

        with gr.Row():
            # Columna izquierda: Entrada y Controles
            with gr.Column(scale=1):
                gr.Markdown("### 📥 1. Fotografía de Archivo")
                entrada_img = gr.Image(
                    type="pil",
                    label="Subir foto escaneada a restaurar",
                    sources=["upload", "clipboard"]
                )

                # Acordeón de Arrugas y Pliegues
                with gr.Accordion("🩹 Remoción de Arrugas y Grietas de Papel", open=True):
                    check_arrugas = gr.Checkbox(
                        label="Eliminar arrugas, pliegues y grietas físicas",
                        value=True,
                        info="Aplica morfología multiescala e Inpainting Telea para borrar grietas sin alterar ojos ni fisonomía."
                    )
                    slider_sensibilidad_arrugas = gr.Slider(
                        minimum=15,
                        maximum=65,
                        value=45,
                        step=5,
                        label="Sensibilidad de detección de grietas",
                        info="Valores recomendados: 40-50 para pliegues y fisuras de emulsión."
                    )
                    radio_inpaint = gr.Radio(
                        choices=["telea", "ns"],
                        value="telea",
                        label="Algoritmo de Inpainting",
                        info="Telea (Marcha Rápida) recomendado por su estabilidad."
                    )

                # Acordeón de Preservación Facial y Difusión
                with gr.Accordion("👤 Preservación Facial y Realce Generativo", open=True):
                    check_proteger_identidad = gr.Checkbox(
                        label="Preservar estrictamente la identidad del rostro",
                        value=True,
                        info="Evita que la IA modifique las facciones, mirada o expresión de la persona retratada."
                    )
                    radio_metodo_pipeline = gr.Radio(
                        choices=[
                            "Pipeline Híbrido (PDI + Difusión)",
                            "Línea Base Clásica (Solo PDI)",
                            "Restauración por Difusión Generativa"
                        ],
                        value="Pipeline Híbrido (PDI + Difusión)",
                        label="Método del Pipeline"
                    )
                    slider_intensidad_ia = gr.Slider(
                        minimum=0.10,
                        maximum=0.45,
                        value=0.25,
                        step=0.05,
                        label="Intensidad de Realce IA (Strength)",
                        info="Valores entre 0.20 y 0.30 restauran nitidez sin deformar rostros."
                    )
                    check_vlm = gr.Checkbox(
                        label="Ejecutar análisis visual semántico con VLM (SmolVLM2)",
                        value=True
                    )

                btn_procesar = gr.Button("✨ Iniciar Restauración y Auditoría", variant="primary", size="lg")

            # Columna derecha: Resultados y Auditoría
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ 2. Resultados Visuales y Comparación")
                
                with gr.Tabs():
                    with gr.TabItem("Vista Restaurada"):
                        salida_restaurada = gr.Image(type="pil", label="Fotografía Restaurada")

                    with gr.TabItem("Máscara de Arrugas / Daño"):
                        salida_mascara_arrugas = gr.Image(
                            type="pil",
                            label="Fisuras y Arrugas Detectadas por el Algoritmo (Blanco = Zona Reparada)"
                        )
                        gr.Markdown("*Transparencia total: podés ver exactamente qué zonas de la foto fueron identificadas como daño físico.*")

                    with gr.TabItem("Mapa de Intervención (Calor)"):
                        salida_heatmap = gr.Image(
                            type="pil",
                            label="Zonas de Mayor Modificación (Colormap INFERNO)"
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
            inputs=[
                entrada_img,
                check_arrugas,
                slider_sensibilidad_arrugas,
                radio_inpaint,
                check_proteger_identidad,
                radio_metodo_pipeline,
                slider_intensidad_ia,
                check_vlm
            ],
            outputs=[
                salida_restaurada,
                salida_mascara_arrugas,
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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=share_public
    )


if __name__ == "__main__":
    main()
