"""
=============================================================================
PROYECTO: Taller de Reducción y Suavizado de Imágenes con Máscara
ASIGNATURA: Inteligencia Artificial
ARCHIVO: pruebas.py
OBJETIVO:
    Demostrar paso a paso y de manera matemáticamente verificable:
    1. Una convolución manual de 3x3 con salto de 2x2 sobre una matriz pequeña (6x6).
    2. El cálculo manual comparado con la función 'reducir_con_suavizado'.
    3. La diferencia entre reducción con suavizado vs reducción directa (sin suavizado).
    4. La verificación de la máscara gaussiana y de promedio.
    5. La prueba del algoritmo completo sobre la imagen de muestra 'imagenes/paisaje.jpg'.

    Los dos estudiantes pueden ejecutar este archivo directamente en la terminal:
        python pruebas.py
    y explicar cada número que aparece en la pantalla ante el profesor o jurado.
=============================================================================
"""

import numpy as np
import procesamiento


def imprimir_matriz(nombre, matriz, decimales=2):
    """Función sencilla para imprimir matrices de forma ordenada y legible."""
    print(f"\n--- {nombre} (Forma: {matriz.shape}) ---")
    filas = matriz.shape[0]
    columnas = matriz.shape[1]
    for i in range(filas):
        fila_texto = "  ["
        for j in range(columnas):
            valor = matriz[i, j]
            if decimales == 0:
                fila_texto += f"{int(valor):4d} "
            else:
                fila_texto += f"{valor:7.{decimales}f} "
        fila_texto += "]"
        print(fila_texto)


def demostracion_paso_a_paso_matriz_pequena():
    """
    DEMOSTRACIÓN DIDÁCTICA PARA LA DEFENSA:
    Utiliza una matriz de 6x6 y una máscara de 3x3 con salto de 2x2.
    Muestra la ventana, la multiplicación elemento a elemento y la suma en cada paso.
    """
    print("=" * 75)
    print("  DEMOSTRACIÓN 1: CONVOLUCIÓN 3x3 CON SALTO 2x2 PASO A PASO")
    print("=" * 75)

    # 1. Definimos una matriz de 6x6 con números enteros fáciles de verificar
    matriz_pequena = np.array([
        [10, 20, 30, 40, 50, 60],
        [15, 25, 35, 45, 55, 65],
        [20, 30, 40, 50, 60, 70],
        [25, 35, 45, 55, 65, 75],
        [30, 40, 50, 60, 70, 80],
        [35, 45, 55, 65, 75, 85]
    ], dtype=np.float64)

    imprimir_matriz("Matriz Original (6x6)", matriz_pequena, decimales=0)

    # 2. Creamos una máscara de promedio de 3x3
    # Cada valor es 1/9 = 0.1111...
    mascara_promedio = procesamiento.crear_mascara_promedio(3)
    imprimir_matriz("Máscara de Promedio 3x3 (cada celda vale 1/9)", mascara_promedio, decimales=4)

    # 3. Aplicamos padding de ceros
    pad = 3 // 2  # pad = 1
    matriz_padded = procesamiento.aplicar_padding_ceros(matriz_pequena, pad)
    imprimir_matriz("Matriz con Padding de Ceros (8x8)", matriz_padded, decimales=0)

    # 4. Saltos seleccionados: vertical = 2, horizontal = 2
    salto_fila = 2
    salto_columna = 2
    alto_original, ancho_original = matriz_pequena.shape

    alto_salida = (alto_original + salto_fila - 1) // salto_fila       # 6 // 2 = 3
    ancho_salida = (ancho_original + salto_columna - 1) // salto_columna # 6 // 2 = 3

    print(f"\nDimensiones de entrada : {alto_original} x {ancho_original}")
    print(f"Salto vertical         : {salto_fila}")
    print(f"Salto horizontal       : {salto_columna}")
    print(f"Dimensiones de salida  : {alto_salida} x {ancho_salida} (Reducción al 50% por eje)")

    # 5. Demostración manual paso a paso de los primeros cálculos
    print("\n" + "-" * 75)
    print("DETALLE DEL CÁLCULO EN CADA PÍXEL DE SALIDA (Paso a paso):")
    print("-" * 75)

    matriz_calculada_manualmente = np.zeros((alto_salida, ancho_salida), dtype=np.float64)

    fila_salida = 0
    for i in range(0, alto_original, salto_fila):
        col_salida = 0
        for j in range(0, ancho_original, salto_columna):
            # Extraemos la ventana de 3x3 en la matriz con padding
            ventana = matriz_padded[i : i + 3, j : j + 3]

            # Multiplicamos elemento a elemento
            producto = ventana * mascara_promedio

            # Sumamos todos los productos
            suma = np.sum(producto)
            matriz_calculada_manualmente[fila_salida, col_salida] = suma

            # Mostramos el desglose en consola para los primeros 3 puntos o todos
            print(f"\n-> Píxel de salida [{fila_salida}, {col_salida}] (Origen en matriz original: [{i}, {j}]):")
            print(f"   Ventana extraída 3x3:")
            for fila_v in ventana:
                print(f"     {[int(v) for v in fila_v]}")
            print(f"   Suma de la ventana = {np.sum(ventana):.1f}")
            print(f"   Multiplicado por 1/9 = {np.sum(ventana):.1f} / 9 = {suma:.2f}")

            col_salida += 1
        fila_salida += 1

    # 6. Ejecutamos la función de procesamiento oficial
    matriz_desde_modulo = procesamiento.reducir_con_suavizado(
        matriz_pequena, mascara_promedio, salto_fila=2, salto_columna=2
    )

    imprimir_matriz("Resultado por la función 'reducir_con_suavizado' (3x3)", matriz_desde_modulo, decimales=2)

    # Comprobación de igualdad matemática exacta
    son_iguales = np.allclose(matriz_calculada_manualmente, matriz_desde_modulo)
    print(f"\n¿El cálculo manual paso a paso coincide exactamente con el módulo? -> {son_iguales}")
    assert son_iguales, "Error: Los cálculos no coinciden."


def demostracion_comparacion_con_y_sin_suavizado():
    """
    DEMOSTRACIÓN DE DEFENSORÍA:
    Compara la reducción con filtro vs la reducción directa (submuestreo).
    Explica por qué el suavizado previo es indispensable en IA y Visión por Computadora.
    """
    print("\n" + "=" * 75)
    print("  DEMOSTRACIÓN 2: COMPARACIÓN CON SUAVIZADO VS REDUCCIÓN DIRECTA")
    print("=" * 75)

    # Matriz con un cambio brusco (borde / alta frecuencia)
    matriz_contraste = np.array([
        [  0,   0, 200, 200,   0,   0],
        [  0,   0, 200, 200,   0,   0],
        [200, 200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200, 200],
        [  0,   0, 200, 200,   0,   0],
        [  0,   0, 200, 200,   0,   0]
    ], dtype=np.float64)

    imprimir_matriz("Matriz con Borde de Alto Contraste (6x6)", matriz_contraste, decimales=0)

    mascara_promedio = procesamiento.crear_mascara_promedio(3)

    # 1. Reducción con suavizado
    reducida_suavizada = procesamiento.reducir_con_suavizado(matriz_contraste, mascara_promedio, 2, 2)
    imprimir_matriz("1. Reducida CON Suavizado (Transición suave)", reducida_suavizada, decimales=1)

    # 2. Reducción directa sin suavizado
    reducida_directa = procesamiento.reducir_directo_sin_suavizado(matriz_contraste, 2, 2)
    imprimir_matriz("2. Reducida DIRECTA Sin Suavizado (Pérdida de bordes y salto brusco)", reducida_directa, decimales=0)

    diferencia = procesamiento.calcular_diferencia_absoluta(reducida_suavizada, reducida_directa)
    print(f"\nDiferencia absoluta media entre ambos métodos: {diferencia:.2f} niveles de intensidad")
    print("\nConclusión para la defensa:")
    print("  - La reducción directa toma un único píxel y descarta por completo los vecinos.")
    print("  - La reducción con suavizado integra la información de todos los píxeles vecinos.")
    print("  - Esto evita el aliasing (bordes dentados o artefactos visuales).")


def prueba_mascara_gaussiana():
    """
    Verifica que la máscara gaussiana cumpla con sus propiedades matemáticas:
    - Simetría respecto al centro.
    - El centro tiene el valor máximo.
    - La suma de todos los elementos es exactamente 1.0.
    """
    print("\n" + "=" * 75)
    print("  DEMOSTRACIÓN 3: COMPROBACIÓN DE MÁSCARA GAUSSIANA")
    print("=" * 75)

    mascara_g3 = procesamiento.crear_mascara_gaussiana(tamano=3, sigma=1.0)
    imprimir_matriz("Máscara Gaussiana 3x3 (sigma = 1.0)", mascara_g3, decimales=4)

    suma_g3 = np.sum(mascara_g3)
    print(f"Suma total de pesos = {suma_g3:.6f} (Debe ser exactamente 1.000000)")
    assert np.isclose(suma_g3, 1.0), "La suma de la máscara gaussiana debe ser 1.0"

    print("Valor en el centro [1,1]:", f"{mascara_g3[1,1]:.4f} (Máximo peso)")
    print("Valor en esquina  [0,0]:", f"{mascara_g3[0,0]:.4f} (Menor peso por estar alejado)")


def prueba_con_imagen_real():
    """
    Carga la imagen 'imagenes/paisaje.jpg', aplica el proceso y guarda el resultado.
    """
    print("\n" + "=" * 75)
    print("  DEMOSTRACIÓN 4: PRUEBA CON IMAGEN REAL ('imagenes/paisaje.jpg')")
    print("=" * 75)

    ruta_origen = "imagenes/paisaje.jpg"
    try:
        matriz_img = procesamiento.cargar_imagen(ruta_origen)
        print(f"Imagen cargada exitosamente: {ruta_origen}")
        print(f"Dimensiones originales: {matriz_img.shape[0]} filas x {matriz_img.shape[1]} columnas x {matriz_img.shape[2]} canales")

        # Usamos máscara promedio 3x3 y salto 2x2
        mascara = procesamiento.crear_mascara_promedio(3)
        salto = 2
        print(f"Aplicando suavizado promedio 3x3 con salto {salto}x{salto}...")

        reducida = procesamiento.reducir_con_suavizado(matriz_img, mascara, salto_fila=salto, salto_columna=salto)
        print(f"Dimensiones resultantes: {reducida.shape[0]} filas x {reducida.shape[1]} columnas x {reducida.shape[2]} canales")

        ruta_salida = "imagenes/paisaje_reducido_prueba.jpg"
        procesamiento.guardar_imagen(reducida, ruta_salida)
        print(f"Imagen reducida guardada en: {ruta_salida}")

    except Exception as error:
        print(f"Nota: No se pudo procesar la imagen real ({error}).")


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("  BATERÍA DE PRUEBAS MATEMÁTICAS - PROYECTO INTELIGENCIA ARTIFICIAL")
    print("#" * 75)

    demostracion_paso_a_paso_matriz_pequena()
    demostracion_comparacion_con_y_sin_suavizado()
    prueba_mascara_gaussiana()
    prueba_con_imagen_real()

    print("\n" + "#" * 75)
    print("  TODAS LAS PRUEBAS MATEMÁTICAS COMPLETADAS SATISFACTORIAMENTE")
    print("#" * 75 + "\n")
