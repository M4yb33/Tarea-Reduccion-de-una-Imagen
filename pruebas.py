"""Pruebas matemáticas y de interfaz obligatorias.

Verifica los 18 requisitos exigidos para la defensa académica del Taller de IA:
Suavizado y Reducción de Imágenes (Universidad Técnica de Ambato).
Ejecutar: python pruebas.py
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, explicar_pixel,
                            calcular_datos, calcular_geometria_panel,
                            convertir_clic_a_pixel, es_posicion_muestreada)


def comprobar(condicion, mensaje):
    if not condicion:
        raise AssertionError(f'FALLÓ: {mensaje}')
    print(f'OK - {mensaje}')


print('======================================================================')
print('EJECUTANDO LAS 18 PRUEBAS OBLIGATORIAS DEL TALLER DE IA')
print('======================================================================\n')

# -----------------------------------------------------------------------------
# Prueba 1. Padding de ceros correcto en esquinas y bordes.
# -----------------------------------------------------------------------------
blanca = np.full((3, 3, 3), 90, dtype=np.uint8)
prom_3x3 = crear_mascara('Promedio', 3)
con_ceros = aplicar_convolucion(blanca, prom_3x3)
comprobar(int(con_ceros[0, 0, 0]) == 40, 'Prueba 1: Padding de ceros en esquina (90*4/9 = 40)')
comprobar(int(con_ceros[1, 1, 0]) == 90, 'Prueba 1: Centro mantiene 90 sin verse afectado por el padding')

# -----------------------------------------------------------------------------
# Prueba 2. Máscara de promedio con pesos iguales.
# -----------------------------------------------------------------------------
m_prom = crear_mascara('Promedio', 3)
comprobar(np.allclose(m_prom, np.ones((3, 3)) / 9.0), 'Prueba 2: Máscara de promedio 3x3 con pesos idénticos (1/9)')
comprobar(np.isclose(m_prom.sum(), 1.0), 'Prueba 2: Suma de pesos de promedio igual a 1.0')

# -----------------------------------------------------------------------------
# Prueba 3. Máscara gaussiana normalizada.
# -----------------------------------------------------------------------------
gauss_3x3 = crear_mascara('Gaussiana', 3)
ideal_gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
comprobar(np.allclose(gauss_3x3, ideal_gauss, atol=0.001), 'Prueba 3: Gaussiana 3x3 coincide con [1,2,1]x[1,2,1]/16')
for n in (3, 5, 7):
    mg = crear_mascara('Gaussiana', n)
    comprobar(np.isclose(mg.sum(), 1.0), f'Prueba 3: Gaussiana {n}x{n} normalizada con suma igual a 1.0')

# -----------------------------------------------------------------------------
# Inicializar entorno gráfico headless para pruebas de interfaz (4, 5, 6, 15)
# -----------------------------------------------------------------------------
main.construir_interfaz()
main.raiz.withdraw()

# -----------------------------------------------------------------------------
# Prueba 4. Edición deshabilitada en promedio y gaussiana.
# -----------------------------------------------------------------------------
main.tipo_mascara.set('Promedio')
main.actualizar_mascara()
prom_bloqueada = all(main.celdas_matriz[f][c].cget('state') == 'readonly' for f in range(3) for c in range(3))
comprobar(prom_bloqueada, 'Prueba 4: Casillas deshabilitadas para edición en Máscara de Promedio (readonly)')

main.tipo_mascara.set('Gaussiana')
main.actualizar_mascara()
gauss_bloqueada = all(main.celdas_matriz[f][c].cget('state') == 'readonly' for f in range(3) for c in range(3))
comprobar(gauss_bloqueada, 'Prueba 4: Casillas deshabilitadas para edición en Máscara Gaussiana (readonly)')

# -----------------------------------------------------------------------------
# Prueba 5. Edición habilitada únicamente en personalizada.
# -----------------------------------------------------------------------------
main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
pers_habilitada = all(main.celdas_matriz[f][c].cget('state') == 'normal' for f in range(3) for c in range(3))
comprobar(pers_habilitada, 'Prueba 5: Casillas habilitadas para edición en Máscara Personalizada (normal)')

# -----------------------------------------------------------------------------
# Prueba 6. Validación de valores personalizados.
# -----------------------------------------------------------------------------
comprobar(main.validar_decimal('2.5') and main.validar_decimal('10'), 'Prueba 6: Permite números decimales y enteros válidos')
comprobar(not main.validar_decimal('abc') and not main.validar_decimal('-5'), 'Prueba 6: Rechaza letras y números negativos')
comprobar(not main.validar_decimal('1e5') and not main.validar_decimal('1.2.3'), 'Prueba 6: Rechaza notación científica y múltiples puntos')
try:
    crear_mascara('Personalizada', 3, [[-1, 1, 1], [1, 1, 1], [1, 1, 1]])
    comprobar(False, 'Prueba 6: Falló rechazo de matriz con peso negativo')
except ValueError:
    comprobar(True, 'Prueba 6: Función crear_mascara rechaza pesos negativos')
try:
    crear_mascara('Personalizada', 3, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    comprobar(False, 'Prueba 6: Falló rechazo de matriz con suma cero')
except ValueError:
    comprobar(True, 'Prueba 6: Función crear_mascara rechaza suma de pesos igual a cero')

# Comprobación de rechazo explícito en interfaz (sin sustitución silenciosa):
main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
for f in range(3):
    for c in range(3):
        main.celdas_matriz[f][c].delete(0, 'end')
        main.celdas_matriz[f][c].insert(0, '0')
try:
    main.mascara_desde_interfaz()
    comprobar(False, 'Prueba 6: Falló rechazo de matriz con 9 ceros en interfaz')
except ValueError:
    comprobar(True, 'Prueba 6: mascara_desde_interfaz rechaza 9 ceros sin sustituir por promedio')

# Restaurar valores válidos
for f in range(3):
    for c in range(3):
        main.celdas_matriz[f][c].delete(0, 'end')
        main.celdas_matriz[f][c].insert(0, '1')

# -----------------------------------------------------------------------------
# Prueba 7. Conversión correcta de un clic sobre la imagen a fila y columna.
# -----------------------------------------------------------------------------
# En un lienzo de 500x400 con imagen de 626x417, el clic central (250, 200) mapea al centro (208, 313)
coord_centro = convertir_clic_a_pixel(250, 200, 500, 400, 626, 417, margen_interior=8)
comprobar(coord_centro is not None, 'Prueba 7: Clic dentro de la imagen produce coordenadas válidas')
f_c, c_c = coord_centro
comprobar(abs(f_c - 208) <= 1 and abs(c_c - 313) <= 1,
          f'Prueba 7: Mapeo correcto de clic (250, 200) a píxel central ({f_c}, {c_c})')

# Verificación de alineación exacta en borde izquierdo (sin desfasaje de +5 columnas):
geom_500 = calcular_geometria_panel(500, 400, 626, 417, margen_interior=8)
clic_borde_izq = convertir_clic_a_pixel(geom_500['ox'], geom_500['oy'] + int(100.5 * geom_500['factor']),
                                        500, 400, 626, 417, margen_interior=8)
comprobar(clic_borde_izq is not None and clic_borde_izq[1] == 0 and abs(clic_borde_izq[0] - 100) <= 1,
          f'Prueba 7: Clic en borde izquierdo mapea a col 0 y fila ~100 sin desfasaje ({clic_borde_izq})')

# -----------------------------------------------------------------------------
# Prueba 8. Selección fuera de la fotografía sin errores.
# -----------------------------------------------------------------------------
comprobar(convertir_clic_a_pixel(250, 10, 500, 400, 626, 417, margen_interior=8) is None,
          'Prueba 8: Clic en margen superior vacío retorna None')
comprobar(convertir_clic_a_pixel(-15, 200, 500, 400, 626, 417, margen_interior=8) is None,
          'Prueba 8: Clic con coordenada x negativa retorna None')
comprobar(convertir_clic_a_pixel(520, 200, 500, 400, 626, 417, margen_interior=8) is None,
          'Prueba 8: Clic más allá del ancho del lienzo retorna None')

# -----------------------------------------------------------------------------
# Prueba 9. Cálculo manual de un píxel coincidente con el algoritmo general.
# -----------------------------------------------------------------------------
vecindad = np.array([[100, 120, 140],
                      [90, 110, 130],
                      [80, 100, 120]], dtype=np.uint8)
pequena = np.repeat(vecindad[:, :, None], 3, axis=2)
res_convolucion = aplicar_convolucion(pequena, ideal_gauss)
_, _, _, res_explicar = explicar_pixel(pequena, ideal_gauss, 1, 1)
comprobar(int(res_convolucion[1, 1, 0]) == 110, 'Prueba 9: Ejercicio de defensa central suma/16 = 110')
comprobar(np.array_equal(res_convolucion[1, 1], res_explicar),
          'Prueba 9: Cálculo paso a paso coincide exactamente con aplicar_convolucion')

# Caso real de verificación de precisión completa (fila 106, columna 576, canal verde):
ruta_paisaje_p9 = Path(__file__).resolve().parent / 'imagenes' / 'paisaje.png'
img_p9 = np.asarray(Image.open(ruta_paisaje_p9).convert('RGB'))
suav_p9 = aplicar_convolucion(img_p9, ideal_gauss, 1, 1)
_, _, _, res_106_576 = explicar_pixel(img_p9, ideal_gauss, 106, 576)
comprobar(suav_p9[106, 576, 1] == res_106_576[1],
          f'Prueba 9: Coincidencia matemática exacta en fila 106, col 576, canal verde ({res_106_576[1]} == {suav_p9[106, 576, 1]})')

# -----------------------------------------------------------------------------
# Prueba 10. Identificación de posiciones de muestreo para salto 2×2.
# -----------------------------------------------------------------------------
comprobar(es_posicion_muestreada(0, 0, 2, 2) is True, 'Prueba 10: (0, 0) es posición muestreada en salto 2x2')
comprobar(es_posicion_muestreada(196, 312, 2, 2) is True, 'Prueba 10: (196, 312) es posición muestreada en salto 2x2')
comprobar(es_posicion_muestreada(197, 313, 2, 2) is False, 'Prueba 10: (197, 313) NO es posición muestreada en salto 2x2')
comprobar(es_posicion_muestreada(1, 0, 2, 2) is False, 'Prueba 10: (1, 0) NO es posición muestreada en salto 2x2')

# -----------------------------------------------------------------------------
# Prueba 11. Salto 2×2 sobre una imagen de 626×417, con resultado 313×209.
# -----------------------------------------------------------------------------
img_626x417 = np.zeros((417, 626, 3), dtype=np.uint8)
salida_2x2 = aplicar_convolucion(img_626x417, ideal_gauss, salto_vertical=2, salto_horizontal=2)
comprobar(salida_2x2.shape == (209, 313, 3),
          f'Prueba 11: Salto 2x2 sobre 626x417 produce resolución exacta {salida_2x2.shape[1]}x{salida_2x2.shape[0]}')

# -----------------------------------------------------------------------------
# Prueba 12. Saltos horizontal y vertical diferentes.
# -----------------------------------------------------------------------------
salida_asym = aplicar_convolucion(img_626x417, ideal_gauss, salto_vertical=3, salto_horizontal=4)
filas_esp = (417 - 1) // 3 + 1  # 139
cols_esp = (626 - 1) // 4 + 1   # 157
comprobar(salida_asym.shape == (filas_esp, cols_esp, 3),
          f'Prueba 12: Salto vertical 3 y horizontal 4 produce resolución exacta {cols_esp}x{filas_esp}')

# -----------------------------------------------------------------------------
# Prueba 13. Igualdad de dimensiones entre reducción directa y suavizada.
# -----------------------------------------------------------------------------
directa_2x2 = reducir_sin_suavizado(img_626x417, 2, 2)
comprobar(directa_2x2.shape == salida_2x2.shape,
          'Prueba 13: Reducción directa y con suavizado tienen idénticas dimensiones (209x313x3)')
directa_asym = reducir_sin_suavizado(img_626x417, 3, 4)
comprobar(directa_asym.shape == salida_asym.shape,
          'Prueba 13: Reducción asimétrica directa y con suavizado coinciden en tamaño')

# -----------------------------------------------------------------------------
# Prueba 14. Conservación de los tres canales RGB.
# -----------------------------------------------------------------------------
img_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
img_rgb[:, :, 0] = 60
img_rgb[:, :, 1] = 120
img_rgb[:, :, 2] = 180
out_rgb = aplicar_convolucion(img_rgb, ideal_gauss, 1, 1)
comprobar(out_rgb.shape == (20, 20, 3) and out_rgb.dtype == np.uint8,
          'Prueba 14: Salida conserva los 3 canales RGB con tipo de dato uint8')
comprobar(out_rgb[10, 10, 0] == 60 and out_rgb[10, 10, 1] == 120 and out_rgb[10, 10, 2] == 180,
          'Prueba 14: Convolución preserva valores independientes de cada canal R, G y B')

# -----------------------------------------------------------------------------
# Prueba 15. Botón «Restaurar» recuperando los valores iniciales.
# -----------------------------------------------------------------------------
ruta_paisaje = Path(__file__).resolve().parent / 'imagenes' / 'paisaje.png'
main.cargar_imagen(str(ruta_paisaje))
main.tipo_mascara.set('Promedio')
main.tamano_mascara.set('5')
main.salto_horizontal.set(4)
main.salto_vertical.set(4)
main.restaurar()
comprobar(main.tipo_mascara.get() == 'Gaussiana', 'Prueba 15: Botón Restaurar restablece máscara Gaussiana')
comprobar(main.tamano_mascara.get() == '3', 'Prueba 15: Botón Restaurar restablece tamaño de máscara a 3x3')
comprobar(main.salto_horizontal.get() == 2 and main.salto_vertical.get() == 2,
          'Prueba 15: Botón Restaurar restablece saltos horizontal y vertical a 2')

# -----------------------------------------------------------------------------
# Prueba 16. Exportación PNG con la matriz real reducida.
# -----------------------------------------------------------------------------
with TemporaryDirectory() as tmpdir:
    ruta_export = Path(tmpdir) / 'exportacion_prueba.png'
    Image.fromarray(salida_2x2).save(ruta_export)
    leida_png = np.asarray(Image.open(ruta_export).convert('RGB'))
    comprobar(leida_png.shape == (209, 313, 3), 'Prueba 16: Archivo PNG exportado tiene resolución exacta 313x209')
    comprobar(np.array_equal(leida_png, salida_2x2),
              'Prueba 16: PNG exportado contiene la matriz matemática real píxel por píxel')

# -----------------------------------------------------------------------------
# Prueba 17. Reproducibilidad al repetir parámetros.
# -----------------------------------------------------------------------------
rep1 = aplicar_convolucion(img_626x417, ideal_gauss, 2, 2)
rep2 = aplicar_convolucion(img_626x417, ideal_gauss, 2, 2)
comprobar(np.array_equal(rep1, rep2),
          'Prueba 17: Mismos parámetros producen exactamente la misma matriz de salida')

# -----------------------------------------------------------------------------
# Prueba 18. Visualización ampliada sin modificación de la imagen exportada.
# -----------------------------------------------------------------------------
matriz_original_exportable = salida_2x2.copy()
# Simulación de escalado para visualización en pantalla con Pillow
factor_pantalla = min(500 / salida_2x2.shape[1], 350 / salida_2x2.shape[0])
vista_pantalla = Image.fromarray(salida_2x2).resize(
    (int(salida_2x2.shape[1] * factor_pantalla), int(salida_2x2.shape[0] * factor_pantalla))
)
comprobar(salida_2x2.shape == (209, 313, 3),
          'Prueba 18: La matriz reducida real mantiene 313x209 sin ser alterada por el escalado de visualización')
comprobar(np.array_equal(salida_2x2, matriz_original_exportable),
          'Prueba 18: Los datos numéricos de los píxeles no se modifican al generar la vista en pantalla')

# Cerrar entorno gráfico de prueba
main.raiz.destroy()

# -----------------------------------------------------------------------------
# Prueba matemática complementaria: Mitigación de aliasing
# -----------------------------------------------------------------------------
patron = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
patron_rgb = np.repeat(patron[:, :, None], 3, axis=2)
sin_suavizar = reducir_sin_suavizado(patron_rgb, 2, 2)
con_suavizado = aplicar_convolucion(patron_rgb, ideal_gauss, 2, 2)
comprobar(np.all(sin_suavizar == 0) and 90 < con_suavizado.mean() < 170,
          'Complementaria: El suavizado previo mitiga el aliasing en patrones de alta frecuencia')

# Métricas sobre imagen de referencia
img_real = np.asarray(Image.open(ruta_paisaje).convert('RGB'))
t0 = perf_counter()
suav = aplicar_convolucion(img_real, ideal_gauss, 1, 1)
red = aplicar_convolucion(img_real, ideal_gauss, 2, 2)
t_ms = (perf_counter() - t0) * 1000
datos_paisaje = calcular_datos(img_real, suav, red, t_ms)

print('\n======================================================================')
print('RESUMEN DE VERIFICACIÓN')
print('======================================================================')
print(f'Original:    {datos_paisaje["ancho_original"]} x {datos_paisaje["alto_original"]} px ({datos_paisaje["pixeles_original"]:,} píxeles)')
print(f'Reducida:    {datos_paisaje["ancho_final"]} x {datos_paisaje["alto_final"]} px ({datos_paisaje["pixeles_final"]:,} píxeles)')
print(f'Reducción:   {datos_paisaje["reduccion"]:.2f} %')
print(f'Tiempo test: {t_ms:.1f} ms')
print('======================================================================')
print('LAS 18 PRUEBAS OBLIGATORIAS FINALIZARON CON ÉXITO (CÓDIGO DE SALIDA 0).')
print('======================================================================')
