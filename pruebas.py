"""Pruebas matemáticas sencillas. Ejecutar: python pruebas.py"""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, explicar_pixel, calcular_datos)


def comprobar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)
    print('OK -', mensaje)


# 1–3. Máscaras, normalización y construcción gaussiana por vectores.
promedio = crear_mascara('Promedio', 3)
comprobar(np.allclose(promedio, np.ones((3, 3)) / 9), 'Promedio 3x3 normalizado')
gaussiana = crear_mascara('Gaussiana', 3)
ideal = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
comprobar(np.allclose(gaussiana, ideal, atol=0.0001), 'Gaussiana 3x3 ~= [1 2 1] exterior /16')
for n in (3, 5, 7):
    comprobar(abs(crear_mascara('Gaussiana', n).sum() - 1) < 1e-10,
              f'Gaussiana {n}x{n}: suma de pesos igual a 1')

# 4. Cálculo manual de la máscara sobre un píxel central.
vecindad = np.array([[100, 120, 140],
                      [90, 110, 130],
                      [80, 100, 120]], dtype=np.uint8)
pequena = np.repeat(vecindad[:, :, None], 3, axis=2)
resultado = aplicar_convolucion(pequena, gaussiana)
comprobar(int(resultado[1, 1, 0]) == 110, 'Cálculo manual central: suma/16 = 110')

# 5. Padding de ceros: media alrededor de esquina cambia por ceros añadidos.
blanca = np.full((3, 3, 3), 90, dtype=np.uint8)
con_ceros = aplicar_convolucion(blanca, promedio)
comprobar(int(con_ceros[0, 0, 0]) == 40, 'Padding de ceros: esquina 90*4/9=40')
comprobar(int(con_ceros[1, 1, 0]) == 90, 'Centro mantiene el valor 90 con padding de ceros')

# 6–8. Saltos y dimensiones calculadas, incluso cuando no son múltiplos.
foto = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
comprobar(aplicar_convolucion(foto, promedio, 1, 1).shape == foto.shape,
          'Salto 1x1 mantiene las dimensiones')
comprobar(aplicar_convolucion(foto, promedio, 2, 2).shape == (3, 4, 3),
          'Salto 2x2: redondeo hacia arriba de 5x7')
comprobar(aplicar_convolucion(foto, promedio, 2, 3).shape == (3, 3, 3),
          'Salto vertical 2 y horizontal 3')

# 9. Propiedad fundamental: convolución reducida = suavizada muestreada.
filtrada = aplicar_convolucion(foto, promedio)
reducida = aplicar_convolucion(foto, promedio, 2, 3)
comprobar(np.array_equal(reducida, filtrada[::2, ::3, :]),
          'Filtrar y saltar da exactamente el mismo resultado')

# 10. Muestreo directo toma posiciones sin modificar intensidad.
directa = reducir_sin_suavizado(foto, 2, 3)
comprobar(np.array_equal(directa, foto[::2, ::3, :]),
          'Reducción directa: solo píxeles originales muestreados')

# 11. Máscara personalizada asimétrica comprueba convolución verdadera.
asimetrica = crear_mascara('Personalizada', 3,
                           [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
resultado_asimetrico = aplicar_convolucion(foto, asimetrica)
comprobar(int(resultado_asimetrico[2, 2, 0]) == int(foto[2, 1, 0]),
          'La máscara se invierte como exige la convolución (no correlación)')

# 12. La explicación de un píxel coincide con el algoritmo completo.
region, pesos, productos, calculado = explicar_pixel(pequena, gaussiana, 1, 1)
comprobar(np.array_equal(calculado, resultado[1, 1]),
          'Detalle didáctico de productos reproduce la salida')

# 13. RGB y uint8 sin valores negativos ni mayores a 255.
comprobar(resultado.dtype == np.uint8 and resultado.shape[2] == 3,
          'Salida de tres canales con enteros de 8 bits')

# 14. Resultados reales de la imagen de ejemplo.
ruta = Path(__file__).resolve().parent / 'imagenes' / 'paisaje.png'
imagen = np.asarray(Image.open(ruta).convert('RGB'))
inicio = perf_counter()
suavizada = aplicar_convolucion(imagen, gaussiana, 1, 1)
mejor = aplicar_convolucion(imagen, gaussiana, 2, 2)
ms = (perf_counter() - inicio) * 1000
datos = calcular_datos(imagen, suavizada, mejor, ms)
comprobar(mejor.shape == (209, 313, 3) and imagen.shape == (417, 626, 3),
          'Paisaje: 626x417 pasa a 313x209 píxeles')
comprobar(datos['pixeles_original'] == 261042 and datos['pixeles_final'] == 65417,
          'Conservación de dimensiones y conteo exacto de píxeles')

# 15. Conservar detalle cuantificable: suavizado reduce variación local.
comprobar(datos['variacion_suavizada'] < datos['variacion_original'],
          'La máscara gaussiana reduce la variación entre píxeles vecinos')

# 16–17. Reproducibilidad y exportación sin reescalado oculto.
comprobar(np.array_equal(mejor, aplicar_convolucion(imagen, gaussiana, 2, 2)),
          'Misma entrada y configuración = misma imagen píxel por píxel')
with TemporaryDirectory() as tmp:
    destino = Path(tmp) / 'salida.png'
    Image.fromarray(mejor).save(destino)
    reabierta = np.asarray(Image.open(destino).convert('RGB'))
    comprobar(np.array_equal(reabierta, mejor), 'PNG exportado sin modificar los píxeles')

# 18. El suavizado reduce el aliasing en un patrón de alta frecuencia.
patron = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
patron_rgb = np.repeat(patron[:, :, None], 3, axis=2)
sin_filtro = reducir_sin_suavizado(patron_rgb, 2, 2)
con_filtro = aplicar_convolucion(patron_rgb, gaussiana, 2, 2)
comprobar(np.all(sin_filtro == 0) and 90 < con_filtro.mean() < 170,
          'Patrón alterno: el filtro evita que todos los cuadros se vuelvan negros')

# 19. Validación de máscaras: impedir tamaños y pesos incorrectos.
try:
    crear_mascara('Personalizada', 3, [[-1, 2, 1]] * 3)
    raise AssertionError('Se esperaba rechazar un peso negativo')
except ValueError:
    print('OK - Mascara de suavizado rechaza pesos negativos')

print(f'\nTodas las pruebas matematicas finalizaron correctamente. Tiempo ejemplo: {ms:.1f} ms.')
print(f'Reducción de píxeles: {datos["reduccion"]:.2f} %.')
