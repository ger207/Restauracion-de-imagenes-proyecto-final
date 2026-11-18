# Bitácora de Vibe Engineering y Colaboración con Agente de IA

**Proyecto:** Restaurador y Analizador de Fotografías Históricas para Archivo Barrial  
**Estudiante (Decision-maker):** Germán Le Piani  
**Agente de IA:** Antigravity (Google DeepMind)  
**Materia:** Procesamiento Digital de Imágenes e Introducción a Visión por Computadora (IFTS 24, 2026)  

---

## 1. Filosofía de Trabajo (Agentic Engineering)
Siguiendo las pautas de Simon Willison (2025):
- **Germán** actúa como ingeniero responsable: define el caso de uso, toma las decisiones arquitectónicas, valida cada componente, aprueba cada diff y asume la autoría y defensa técnica del proyecto.
- **El Agente** asiste en la implementación acelerada de código modular, redacción de documentación exhaustiva y diseño de pruebas.

---

## 2. Registro de Decisiones de Diseño

| Decisión | Quién la tomó | Justificación Técnica / HAI |
|---|---|---|
| **Elección de Sub-tarea C (Fotos Antiguas)** | Germán | Es el escenario que mejor expone la tensión entre *lo real* (luminancia original) y *lo inventado* (color y micro-textura generados). |
| **Definición de Persona Usuaria (Elena)** | Germán | Permite diseñar una interfaz compasiva y honesta para archivistas aficionados, con advertencias claras sobre no usar IA como prueba histórica. |
| **Análisis Visual con VLM liviano** | Germán | En lugar de métricas puramente matemáticas que pueden engañar, un modelo VLM (SmolVLM2) describe en lenguaje natural qué objetos y alteraciones detecta. |
| **Estrategia Híbrida (Local Gradio + Colab T4)** | Consenso (Propuesta Agente, aprobada por Germán) | Garantiza reproducibilidad en equipos con CPU/GPU integrada y acceso a aceleración completa en Colab sin costo. |
| **Inclusión de Línea Base Clásica (PDI)** | Consenso | Demuestra valor comparativo entre algoritmos tradicionales (CLAHE, OpenCV Inpainting) y modelos de difusión generativa. |

---

## 3. Registro de Iteraciones
- **Iteración 1:** Configuración de entorno reproducible con `uv`, Python 3.12, `.gitignore` seguro y scaffolding de módulos.
- **Iteración 2:** Implementación de procesamiento digital clásico (`baseline.py`), mapa de diferencias y honestidad (`honesty.py`).
- **Iteración 3:** Integración del pipeline de difusión (`diffusion.py`) y análisis VLM (`analysis.py`).
- **Iteración 4:** Construcción de la interfaz interactiva Gradio (`app.py`), pruebas con imágenes históricas de muestra y notebook de Colab.
- **Iteración 5:** Documentación final en `README.md` y preparación del guion para el video demo de 5 minutos.
