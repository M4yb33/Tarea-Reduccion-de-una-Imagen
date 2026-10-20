"""Suavizado y reducción de imágenes: máscaras y saltos programados manualmente.

No se utilizan cv2.resize, PIL.resize, GaussianBlur, filter2D ni otras funciones
que realicen el filtrado o la reducción de forma automática.
El padding con ceros se aplica internamente en las operaciones de convolución.
"""

import csv
import os
from datetime import datetime
import numpy as np


def crear_mascara(tipo, tamano, valores=None):
    """Construye una máscara normalizada de dimensiones impares."""
    if tamano not in (3, 5, 7):
        raise ValueError('La máscara debe ser de 3x3, 5x5 o 7x7.')

    if tipo == 'Promedio':
        mascara = np.ones((tamano, tamano), dtype=np.float64)
    elif tipo == 'Gaussiana':
        if tamano == 3:
            # Gaussiana 3x3 exacta de clase: producto exterior [1, 2, 1] x [1, 2, 1] / 16.
            vector = np.array([1.0, 2.0, 1.0], dtype=np.float64)
            mascara = np.outer(vector, vector)
        elif tamano == 5:
            # Gaussiana 5x5 discreta binomial: [1, 4, 6, 4, 1]
            vector = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
            mascara = np.outer(vector, vector)
        else:
            sigma = 1.40
            posiciones = np.arange(tamano) - tamano // 2
            vector = np.exp(-(posiciones ** 2) / (2 * sigma ** 2))
            mascara = np.outer(vector, vector)
    elif tipo == 'Personalizada':
        if valores is None:
            raise ValueError('Introduzca los valores de la máscara personalizada.')
        mascara = np.asarray(valores, dtype=np.float64)
        if mascara.shape != (tamano, tamano):
            raise ValueError(f'La máscara debe contener {tamano} filas y columnas.')
        if not np.all(np.isfinite(mascara)):
            raise ValueError('La máscara no admite valores infinitos o vacíos.')
        if np.any(mascara < 0) or not np.any(mascara > 0):
            raise ValueError('Para suavizar, utilice pesos no negativos y alguno positivo.')
    else:
        raise ValueError('Seleccione Promedio, Gaussiana o Personalizada.')

    return mascara / mascara.sum()


def aplicar_convolucion(imagen, mascara, salto_vertical=1, salto_horizontal=1):
    """Multiplica máscara por vecindades RGB, suma productos y avanza por saltos.

    Se recorren las FILAS y COLUMNAS de la máscara mediante dos bucles sencillos.
    Aplica internamente padding de ceros en los bordes para centrar el kernel.
    No usa funciones de convolución ni de reducción prediseñadas.
    """
    if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
        raise ValueError('Se espera una imagen RGB (alto, ancho, 3) de tipo uint8.')
    if not isinstance(salto_vertical, (int, np.integer)) or salto_vertical < 1:
        raise ValueError('El salto vertical debe ser un entero positivo.')
    if not isinstance(salto_horizontal, (int, np.integer)) or salto_horizontal < 1:
        raise ValueError('El salto horizontal debe ser un entero positivo.')
    mascara = np.asarray(mascara, dtype=np.float64)
    tamano = mascara.shape[0]
    if mascara.ndim != 2 or mascara.shape != (tamano, tamano) or tamano % 2 != 1:
        raise ValueError('La máscara debe ser una matriz cuadrada impar.')
    if not np.all(np.isfinite(mascara)):
        raise ValueError('La máscara contiene valores no válidos.')

    alto, ancho = imagen.shape[:2]
    relleno = tamano // 2
    # Padding de ceros interno obligatorio con precisión float64
    ampliada = np.pad(imagen.astype(np.float64),
                      ((relleno, relleno), (relleno, relleno), (0, 0)), mode='constant')

    filas_salida = (alto - 1) // salto_vertical + 1
    columnas_salida = (ancho - 1) // salto_horizontal + 1
    salida = np.zeros((filas_salida, columnas_salida, 3), dtype=np.float64)

    # Convolución verdadera: invertir los pesos de la máscara. Para las
    # máscaras simétricas (promedio y gaussiana), esta inversión es idéntica.
    invertida = mascara[::-1, ::-1]
    for fila_mascara in range(tamano):
        for columna_mascara in range(tamano):
            peso = float(invertida[fila_mascara, columna_mascara])
            if peso == 0:
                continue
            # Todas las vecindades cuyos centros siguen los saltos elegidos.
            region = ampliada[
                fila_mascara:fila_mascara + filas_salida * salto_vertical:salto_vertical,
                columna_mascara:columna_mascara + columnas_salida * salto_horizontal:salto_horizontal,
                :
            ]
            salida += region * peso  # Producto elemento a elemento + suma.

    return np.clip(np.rint(salida), 0, 255).astype(np.uint8)


def reducir_sin_suavizado(imagen, salto_vertical, salto_horizontal):
    """Referencia experimental: muestrea píxeles sin promediarlos."""
    return imagen[::salto_vertical, ::salto_horizontal, :].copy()


def calcular_datos(original, suavizada, reducida, tiempo_ms):
    """Calcula medidas descriptivas de las imágenes, sin juicios automáticos."""
    alto, ancho = original.shape[:2]
    alto_nuevo, ancho_nuevo = reducida.shape[:2]
    pixeles_original = alto * ancho
    pixeles_final = alto_nuevo * ancho_nuevo

    # Diferencia promedio entre vecinos: mide variación espacial (no es PSNR).
    def variacion(imagen):
        matriz = imagen.astype(np.float32)
        vertical = np.abs(matriz[1:, :, :] - matriz[:-1, :, :]).mean() if matriz.shape[0] > 1 else 0
        horizontal = np.abs(matriz[:, 1:, :] - matriz[:, :-1, :]).mean() if matriz.shape[1] > 1 else 0
        return float((vertical + horizontal) / 2)

    return {
        'original': f'{ancho} × {alto}',
        'final': f'{ancho_nuevo} × {alto_nuevo}',
        'ancho_original': ancho,
        'alto_original': alto,
        'ancho_final': ancho_nuevo,
        'alto_final': alto_nuevo,
        'pixeles_original': pixeles_original,
        'pixeles_final': pixeles_final,
        'reduccion': 100 * (1 - pixeles_final / pixeles_original),
        'media_original': float(original.mean()),
        'media_suavizada': float(suavizada.mean()),
        'variacion_original': variacion(original),
        'variacion_suavizada': variacion(suavizada),
        'tiempo_ms': float(tiempo_ms),
    }


def exportar_metricas_csv(ruta_csv, nombre_imagen, datos, tipo_mascara, tamano_mascara,
                           salto_h, salto_v):
    """Escribe las métricas comprobables en un archivo CSV."""
    encabezados = [
        'Fecha y Hora', 'Imagen', 'Ancho Original', 'Alto Original', 'Pixeles Original',
        'Ancho Final', 'Alto Final', 'Pixeles Final', 'Reduccion (%)',
        'Tipo Mascara', 'Tamano Mascara', 'Salto Horizontal', 'Salto Vertical',
        'Tiempo (ms)', 'Media Original', 'Media Suavizada',
        'Variacion Original', 'Variacion Suavizada'
    ]

    fila = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        nombre_imagen,
        datos.get('ancho_original', ''),
        datos.get('alto_original', ''),
        datos.get('pixeles_original', ''),
        datos.get('ancho_final', ''),
        datos.get('alto_final', ''),
        datos.get('pixeles_final', ''),
        f"{datos.get('reduccion', 0):.2f}",
        tipo_mascara,
        f"{tamano_mascara}x{tamano_mascara}",
        salto_h,
        salto_v,
        f"{datos.get('tiempo_ms', 0):.2f}",
        f"{datos.get('media_original', 0):.2f}",
        f"{datos.get('media_suavizada', 0):.2f}",
        f"{datos.get('variacion_original', 0):.2f}",
        f"{datos.get('variacion_suavizada', 0):.2f}"
    ]

    archivo_existe = os.path.exists(ruta_csv)
    with open(ruta_csv, mode='a', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        if not archivo_existe:
            escritor.writerow(encabezados)
        escritor.writerow(fila)


def explicar_pixel(imagen, mascara, fila, columna):
    """Desglosa cada multiplicación del píxel solicitado para la defensa."""
    tamano = mascara.shape[0]
    radio = tamano // 2
    # Padding de ceros interno
    ampliada = np.pad(imagen, ((radio, radio), (radio, radio), (0, 0)), mode='constant')
    region = ampliada[fila:fila + tamano, columna:columna + tamano]
    pesos = np.asarray(mascara)[::-1, ::-1]
    productos = region.astype(np.float64) * pesos[:, :, None]
    resultado = np.clip(np.rint(productos.sum(axis=(0, 1))), 0, 255).astype(np.uint8)
    return region, pesos, productos, resultado


def calcular_geometria_panel(lienzo_ancho, lienzo_alto, img_ancho, img_alto, margen_interior=8):
    """Calcula dimensiones escaladas, centrado y márgenes de la imagen en el lienzo."""
    if lienzo_ancho <= 0 or lienzo_alto <= 0 or img_ancho <= 0 or img_alto <= 0:
        return None
    lw = max(lienzo_ancho - margen_interior, 1)
    lh = max(lienzo_alto - margen_interior, 1)
    factor = min(lw / img_ancho, lh / img_alto)
    vw = max(1, int(img_ancho * factor))
    vh = max(1, int(img_alto * factor))
    cx = lienzo_ancho // 2
    cy = lienzo_alto // 2
    ox = cx - vw // 2
    oy = cy - vh // 2
    return {
        'factor': factor,
        'vw': vw,
        'vh': vh,
        'cx': cx,
        'cy': cy,
        'ox': ox,
        'oy': oy
    }


def convertir_clic_a_pixel(clic_x, clic_y, lienzo_ancho, lienzo_alto, img_ancho, img_alto, margen_interior=8):
    """Convierte coordenadas del lienzo a fila y columna en la matriz de la imagen."""
    geom = calcular_geometria_panel(lienzo_ancho, lienzo_alto, img_ancho, img_alto, margen_interior)
    if geom is None:
        return None
    ox, oy = geom['ox'], geom['oy']
    vw, vh = geom['vw'], geom['vh']
    factor = geom['factor']

    if clic_x < ox or clic_x >= ox + vw:
        return None
    if clic_y < oy or clic_y >= oy + vh:
        return None

    columna = int((clic_x - ox) / factor)
    fila = int((clic_y - oy) / factor)
    columna = max(0, min(columna, img_ancho - 1))
    fila = max(0, min(fila, img_alto - 1))
    return fila, columna


def es_posicion_muestreada(fila, columna, salto_v, salto_h):
    """Determina si un pixel (fila, columna) es muestreado según los saltos."""
    return (fila % salto_v == 0) and (columna % salto_h == 0)

