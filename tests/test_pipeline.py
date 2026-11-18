"""
Pruebas Unitarias y de Integración del Pipeline.
Verifica:
1. Métodos clásicos de PDI (CLAHE, remoción de arrugas multiescala, inpainting, bilateral).
2. Colorización en espacio L*a*b* (estilos realista, sepia y monocromo).
3. Cálculo de métricas y mapa de calor de intervención residual.
4. Estampado de sello archivístico de procedencia.
"""

import unittest
import numpy as np
from PIL import Image

from src.baseline import (
    ecualizar_clahe,
    remover_arrugas_e_inpainting,
    filtro_bilateral_limpieza,
    pipeline_restauracion_clasica
)
from src.colorization import colorizar_foto_lab
from src.honesty import (
    es_imagen_monocroma,
    calcular_mapa_intervencion,
    generar_informe_honestidad_md,
    estampar_sello_archivistico
)
from src.analysis import informe_completo_vlm


class TestPipelineRestauracion(unittest.TestCase):

    def setUp(self):
        # Crear imagen sintética de prueba (256x256)
        arr = np.full((256, 256, 3), 120, dtype=np.uint8)
        # Rayón artificial brillante
        arr[50:200, 100:104] = 250
        self.img_test = Image.fromarray(arr)

    def test_clahe_conserva_dimensiones(self):
        res = ecualizar_clahe(self.img_test)
        self.assertEqual(res.size, self.img_test.size)
        self.assertEqual(res.mode, "RGB")

    def test_remocion_arrugas_e_inpainting(self):
        res, mascara = remover_arrugas_e_inpainting(self.img_test, sensibilidad=40)
        self.assertEqual(res.size, self.img_test.size)
        self.assertEqual(mascara.size, self.img_test.size)
        # La máscara debe haber detectado el rayón
        arr_mask = np.array(mascara)
        self.assertGreater(np.sum(arr_mask > 0), 0)

    def test_colorizacion_lab(self):
        # Imagen blanco y negro
        img_bn = Image.fromarray(np.full((100, 100, 3), 140, dtype=np.uint8))
        
        # Colorizado realista
        color_real = colorizar_foto_lab(img_bn, estilo="realista", saturacion=0.8)
        self.assertEqual(color_real.size, img_bn.size)
        arr_cr = np.array(color_real)
        # Los canales R, G, B ya no son idénticos
        self.assertFalse(np.array_equal(arr_cr[:, :, 0], arr_cr[:, :, 2]))

        # Sepia
        color_sepia = colorizar_foto_lab(img_bn, estilo="sepia", saturacion=0.8)
        self.assertEqual(color_sepia.size, img_bn.size)

        # Monocromo conserva R==G==B
        color_mono = colorizar_foto_lab(img_bn, estilo="monocromo")
        arr_mono = np.array(color_mono)
        self.assertTrue(np.array_equal(arr_mono[:, :, 0], arr_mono[:, :, 1]))

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

    def test_preservacion_identidad(self):
        from src.diffusion import preservar_identidad_rostro
        res = preservar_identidad_rostro(self.img_test, self.img_test)
        self.assertEqual(res.size, self.img_test.size)

    def test_informe_completo_vlm(self):
        informe = informe_completo_vlm(self.img_test, self.img_test, {})
        self.assertIn("Análisis de Contenido y Semántica", informe)
        self.assertIn("Verificación de Objetos y Estructura", informe)
        self.assertNotIn("Evaluación del Modelo de Visión y Lenguaje", informe)


if __name__ == "__main__":
    unittest.main()
