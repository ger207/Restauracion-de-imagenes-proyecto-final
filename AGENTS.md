# Bitácora de Vibe Engineering y Colaboración con Agente de IA

**Proyecto:** Restaurador y Analizador de Fotografías Históricas para Archivo Barrial  
**Estudiante (Decision-maker):** Germán Le Piani  
**Agente de IA:** Antigravity (Google DeepMind)  
**Materia:** Procesamiento Digital de Imágenes e Introducción a Visión por Computadora (IFTS 24, 2026)  

---

## 1. Filosofía de Trabajo (Agentic Engineering)
Siguiendo las pautas de Simon Willison (2025):
- **Germán** actúa como ingeniero responsable: define el caso de uso, prueba los prototipos en Colab con GPU y en local, identifica deficiencias visuales y deformaciones faciales, toma las decisiones arquitectónicas de simplificación y aprueba cada cambio.
- **El Agente** asiste en la implementación acelerada de código modular, optimización de algoritmos de PDI, redacción de documentación y diseño de pruebas.

---

## 2. Registro de Decisiones de Diseño

| Decisión | Quién la tomó | Justificación Técnica / HAI |
|---|---|---|
| **Elección de Sub-tarea C (Fotos Antiguas)** | Germán | Es el escenario que mejor expone la tensión entre *lo real* (luminancia original) y *lo inventado* (detalles sintetizados). |
| **Definición de Persona Usuaria (Elena)** | Germán | Permite diseñar una interfaz compasiva y honesta para archivistas aficionados, con advertencias claras sobre no usar IA como prueba histórica. |
| **Análisis Visual con VLM liviano** | Germán | En lugar de métricas puramente matemáticas que pueden engañar, un modelo VLM (SmolVLM2) describe en lenguaje natural qué objetos y alteraciones detecta. |
| **Estrategia Híbrida (Local Gradio + Colab T4)** | Consenso | Garantiza reproducibilidad en equipos con CPU/GPU integrada y acceso a aceleración completa en Colab sin costo. |
| **Remoción Interactiva de Arrugas y Pliegues** | **Germán** | La prueba empírica con retratos reales demostró que las grietas gruesas no se borraban; se incorporó detección Top-Hat e Inpainting Telea con blindaje ocular. |
| **Descarte de Colorización (Fidelidad Monocroma)** | **Germán** | Se eliminó la colorización por generar artefactos y desvirtuar el rigor documental; se preserva el 100% del blanco y negro histórico auténtico. |
| **Preservación Estricta de Identidad Facial** | **Germán** | Al probar en Colab con difusión, se detectó que la IA alteraba sutilmente las caras. Se implementó anclaje facial anatómico con detector LBP para congelar la fisonomía original. |
| **Simplificación de Interfaz (Sin Muestras)** | **Germán** | Eliminación de galerías de ejemplo precargadas para un flujo de trabajo enfocado directamente en la foto que el archivista sube. |

---

## 3. Registro de Iteraciones
- **Iteración 1:** Configuración de entorno reproducible con `uv`, Python 3.12, `.gitignore` seguro y scaffolding de módulos.
- **Iteración 2:** Implementación de procesamiento digital clásico (`baseline.py`), mapa de diferencias y honestidad (`honesty.py`).
- **Iteración 3:** Integración del pipeline de difusión (`diffusion.py`) y análisis VLM (`analysis.py`).
- **Iteración 4:** Pruebas con retratos dañados y optimización de memoria.
- **Iteración 5:** Corrección de detección de arrugas por componentes conectados y blindaje LBP en ojos.
- **Iteración 6:** Pruebas en Google Colab con GPU T4. Simplificación de la UI (remoción de muestras precargadas), eliminación de colorización para preservar fidelidad histórica monocroma, e implementación del anclaje de identidad facial en difusión para evitar deformaciones en los retratos.
