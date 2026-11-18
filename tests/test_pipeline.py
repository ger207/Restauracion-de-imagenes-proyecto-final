"""
Pruebas Unitarias y de Integración del Pipeline.
Verifica:
1. Métodos clásicos de PDI (CLAHE, inpainting, filtrado bilateral).
2. Cálculo de métricas y mapa de calor de intervención residual.
3. Estampado de sello archivístico de procedencia.
4. Orquestación general del flujo.
"""

import os
import unittest
import numpy as np
from PIL import Image

from src.baseline import (
    ecualizar_clahe,
    inpainting_clasico,
    filtro_bilateral_limpieza,
    pipeline_restauracion_clasica
)
from src.honesty import (
    es_imagen_monocroma,
    calcular_mapa_intervencion,
    generar_informe_honestidad_md,
    estampar_sello_archivistico
)
from src.analysis import informe_completo_vlm


class TestPipelineRestauracion(unittest.TestCase):

    def setUp(self):
        # Crear imagen sintética de prueba (512x512)
        arr = np.full((256, 256, 3), 120, dtype=np.uint8)
        # Rayón artificial
        arr[50:200, 100:104] = 250
        self.img_test = Image.fromarray(arr)

    def test_clahe_conserva_dimensiones(self):
        res = ecualizar_clahe(self.img_test)
        self.assertEqual(res.size, self.img_test.size)
        self.assertEqual(res.mode, "RGB")

    def test_inpainting_clasico(self):
        res, mascara = inpainting_clasico(self.img_test)
        self.assertEqual(res.size, self.img_test.size)
        self.assertEqual(mascara.size, self.img_test.size)

    def test_filtro_bilateral(self):
        res = filtro_bilateral_limpieza(self.img_test)
        self.assertEqual(res.size, self.img_test.size)

    def test_deteccion_monocromia(self):
        img_bn = Image.fromarray(np.full((100, 100, 3), 150, dtype=np.uint8))
        self.assertTrue(es_imagen_monocroma(img_bn))

        img_color = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img_color.paste((255, 0, 0), [0, 0, 50, 50])
        self.assertFalse(es_imagen_monocroma(img_color))

    def test_mapa_intervencion_y_metricas(self):
        img_modificada = self.img_test.copy()
        # Modificar un área
        arr = np.array(img_modificada)
        arr[50:100, 50:100] = [200, 50, 50]
        img_mod = Image.fromarray(arr)

        heatmap, metricas = calcular_mapa_intervencion(self.img_test, img_mod)
        self.assertEqual(heatmap.size, self.img_test.size)
        self.assertIn("porcentaje_modificado", metricas)
        self.assertIn("ssim", metricas)
        self.assertIn("psnr_db", metricas)
        self.assertGreater(metricas["porcentaje_modificado"], 0.0)

        informe = generar_informe_honestidad_md(metricas)
        self.assertIn("Auditoría de Fidelidad Histórica", informe)

    def test_sello_archivistico(self):
        res = estampar_sello_archivistico(self.img_test)
        self.assertGreater(res.size[1], self.img_test.size[1])

    def test_informe_vlm(self):
        informe = informe_completo_vlm(self.img_test, self.img_test, {"porcentaje_modificado": 5.0})
        self.assertIn("Análisis de Contenido y Semántica", informe)


if __name__ == "__main__":
    unittest.main()
