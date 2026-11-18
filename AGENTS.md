# Bitácora de Vibe Engineering y Colaboración con Agente de IA

**Proyecto:** Restaurador y Analizador de Fotografías Históricas para Archivo Barrial  
**Estudiante (Decision-maker):** Germán Le Piani  
**Agente de IA:** Antigravity (Google DeepMind)  
**Materia:** Procesamiento Digital de Imágenes e Introducción a Visión por Computadora (IFTS 24, 2026)  

---

## 1. Filosofía de Trabajo (Agentic Engineering)
Siguiendo las pautas de Simon Willison (2025):
- **Germán** actúa como ingeniero responsable: define el caso de uso, prueba los prototipos con imágenes reales de alta exigencia, identifica deficiencias visuales, toma las decisiones arquitectónicas y aprueba cada cambio.
- **El Agente** asiste en la implementación acelerada de código modular, optimización de algoritmos de PDI, redacción de documentación y diseño de pruebas.

---

## 2. Registro de Decisiones de Diseño

| Decisión | Quién la tomó | Justificación Técnica / HAI |
|---|---|---|
| **Elección de Sub-tarea C (Fotos Antiguas)** | Germán | Es el escenario que mejor expone la tensión entre *lo real* (luminancia original) y *lo inventado* (color y micro-textura generados). |
| **Definición de Persona Usuaria (Elena)** | Germán | Permite diseñar una interfaz compasiva y honesta para archivistas aficionados, con advertencias claras sobre no usar IA como prueba histórica. |
| **Análisis Visual con VLM liviano** | Germán | En lugar de métricas puramente matemáticas que pueden engañar, un modelo VLM (SmolVLM2) describe en lenguaje natural qué objetos y alteraciones detecta. |
| **Estrategia Híbrida (Local Gradio + Colab T4)** | Consenso | Garantiza reproducibilidad en equipos con CPU/GPU integrada y acceso a aceleración completa en Colab sin costo. |
| **Remoción Interactiva de Arrugas y Pliegues** | **Germán** | La prueba empírica con retratos reales demostró que las grietas gruesas no se borraban; se incorporó detección multiescala Top-Hat/Black-Hat e Inpainting Telea/NS con control de sensibilidad. |
| **Módulo de Colorización a Demanda (L\*a\*b\*)** | **Germán** | El usuario debe tener la potestad de decidir si colorear (realista/sepia) o preservar la fidelidad del blanco y negro original de la fotografía. |

---

## 3. Registro de Iteraciones
- **Iteración 1:** Configuración de entorno reproducible con `uv`, Python 3.12, `.gitignore` seguro y scaffolding de módulos.
- **Iteración 2:** Implementación de procesamiento digital clásico (`baseline.py`), mapa de diferencias y honestidad (`honesty.py`).
- **Iteración 3:** Integración del pipeline de difusión (`diffusion.py`) y análisis VLM (`analysis.py`).
- **Iteración 4:** Detección de cuello de botella en CPU con imágenes de prueba severamente dañadas. Incorporación de remoción profunda de arrugas (`remover_arrugas_e_inpainting`), módulo de colorización paramétrica (`colorization.py`) y visualización de máscara de grietas en Gradio.
- **Iteración 5:** Documentación final en `README.md`, suite de pruebas unitarias al 100% y preparación para video demo.
