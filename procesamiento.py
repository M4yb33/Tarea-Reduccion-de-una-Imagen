"""
=============================================================================
PROYECTO: Taller de Reducción y Suavizado de Imágenes con Máscara
ASIGNATURA: Inteligencia Artificial
DESCRIPCIÓN:
    Este archivo contiene únicamente las funciones matemáticas necesarias para:
    1. Cargar y guardar imágenes como matrices NumPy.
    2. Crear máscaras de suavizado (Promedio y Gaussiana).
    3. Aplicar padding de ceros a la imagen.
    4. Reducir la imagen aplicando convolución manual y saltos (stride).
    5. Reducir la imagen de forma directa (sin suavizado) para comparar.

    Todas las funciones están escritas en Python básico y NumPy, usando bucles
    explícitos 'for' para que el procedimiento sea fácil de seguir y defender.
=============================================================================
"""

import math
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. CARGA Y GUARDADO DE IMÁGENES
# ---------------------------------------------------------------------------

def cargar_imagen(ruta_archivo):
    """
    Carga un archivo de imagen y lo convierte en una matriz de NumPy.

    ¿Qué recibe?
        - ruta_archivo (str): Ruta en el disco donde se encuentra la imagen.

    ¿Qué operación realiza?
        - Abre la imagen con la librería PIL.
        - La convierte a formato RGB (3 canales: Rojo, Verde, Azul).
        - La transforma en un arreglo NumPy de tipo float64 (valores entre 0.0 y 255.0).
          Se usa float64 para evitar errores de desbordamiento en las operaciones matemáticas.

    ¿Por qué es necesaria?
        - Las operaciones de convolución y filtrado requieren trabajar con matrices numéricas.

    ¿Qué devuelve?
        - matriz_imagen (np.ndarray): Matriz tridimensional con dimensiones (alto, ancho, 3).
    """
    imagen_pil = Image.open(ruta_archivo)
    imagen_rgb = imagen_pil.convert("RGB")
    matriz_imagen = np.array(imagen_rgb, dtype=np.float64)
    return matriz_imagen


def guardar_imagen(matriz_imagen, ruta_archivo):
    """
    Guarda una matriz de imagen como archivo en el disco (JPEG o PNG).

    ¿Qué recibe?
        - matriz_imagen (np.ndarray): Matriz numérica de la imagen.
        - ruta_archivo (str): Ruta de destino para guardar la imagen.

    ¿Qué operación realiza?
        - Limita los valores al rango válido [0, 255] usando np.clip.
        - Convierte los valores de flotante a enteros de 8 bits (uint8).
        - Guarda la imagen en disco utilizando PIL.

    ¿Por qué es necesaria?
        - Permite exportar y verificar visualmente los resultados obtenidos.

    ¿Qué devuelve?
        - None.
    """
    matriz_clipeada = np.clip(matriz_imagen, 0, 255)
    matriz_uint8 = matriz_clipeada.astype(np.uint8)
    imagen_pil = Image.fromarray(matriz_uint8)
    imagen_pil.save(ruta_archivo)


# ---------------------------------------------------------------------------
# 2. CREACIÓN DE MÁSCARAS DE SUAVIZADO
# ---------------------------------------------------------------------------

def crear_mascara_promedio(tamano):
    """
    Crea una máscara de suavizado por promedio (filtro de caja / Box Blur).

    ¿Qué recibe?
        - tamano (int): Tamaño del kernel o ventana (debe ser impar, ej. 3, 5, 7).

    ¿Qué operación matemática realiza?
        - Cada celda de la matriz recibe el valor:
              M[i, j] = 1 / (tamano * tamano)
        - Todos los coeficientes son idénticos y la suma total de la matriz es 1.0.

    ¿Por qué es necesaria?
        - Calcula el promedio aritmético de los píxeles vecinos. Al promediar,
          se atenúan las frecuencias altas (ruido y cambios bruscos), lo que
          evita el efecto de aliasing al reducir la imagen.
        - La suma de los pesos debe ser 1.0 para que el brillo general no cambie.

    ¿Qué devuelve?
        - mascara (np.ndarray): Matriz bidimensional de (tamano, tamano) con tipo float64.

    ¿Qué ocurre si se modifica el parámetro 'tamano'?
        - A mayor tamaño (ej. 5 o 7), mayor cantidad de vecinos se promedian,
          produciendo un desenfoque más pronunciado.
    """
    total_celdas = tamano * tamano
    valor_peso = 1.0 / total_celdas

    # Creamos una matriz llena con el mismo peso en cada posición
    mascara = np.full((tamano, tamano), valor_peso, dtype=np.float64)
    return mascara


def crear_mascara_gaussiana(tamano, sigma=1.0):
    """
    Crea una máscara de suavizado basada en la distribución Gaussiana 2D.

    ¿Qué recibe?
        - tamano (int): Tamaño impar de la máscara (ej. 3, 5, 7).
        - sigma (float): Desviación estándar de la función gaussiana (por defecto 1.0).

    ¿Qué operación matemática realiza?
        - Para cada celda (i, j), calcula la distancia respecto al centro de la máscara:
              x = j - radio
              y = i - radio
        - Aplica la fórmula gaussiana bidimensional:
              G(x, y) = exp( -(x^2 + y^2) / (2 * sigma^2) )
        - Finalmente, normaliza la matriz dividiendo cada celda entre la suma de todas
          las celdas, garantizando que la suma total sea exactamente 1.0:
              M[i, j] = G(x, y) / suma_total

    ¿Por qué es necesaria?
        - A diferencia del promedio, la función gaussiana asigna mayor importancia
          al píxel central y menor importancia a los píxeles alejados. Esto produce
          un suavizado más suave y natural sin perder tanta definición de bordes.

    ¿Qué devuelve?
        - mascara (np.ndarray): Matriz bidimensional de (tamano, tamano) normalizada a suma 1.0.

    ¿Qué ocurre si se modifica 'sigma'?
        - Un sigma pequeño (ej. 0.5) concentra el peso en el centro (menos desenfoque).
        - Un sigma grande (ej. 2.0) dispersa el peso uniformemente (más desenfoque).
    """
    radio = tamano // 2
    mascara = np.zeros((tamano, tamano), dtype=np.float64)

    # Recorremos cada posición de la máscara con bucles for
    for i in range(tamano):
        for j in range(tamano):
            x = j - radio
            y = i - radio
            exponente = -(x ** 2 + y ** 2) / (2.0 * (sigma ** 2))
            mascara[i, j] = math.exp(exponente)

    # Normalización: dividimos para que la suma de todos los pesos sea 1.0
    suma_total = np.sum(mascara)
    mascara = mascara / suma_total

    return mascara


def normalizar_mascara(mascara):
    """
    Normaliza cualquier matriz (útil para máscaras personalizadas).

    ¿Qué recibe?
        - mascara (np.ndarray): Matriz bidimensional con pesos numéricos.

    ¿Qué operación matemática realiza?
        - Divide cada elemento entre la suma total de los elementos.
        - Si la suma es cero, devuelve la matriz sin cambios para evitar división por cero.

    ¿Por qué es necesaria?
        - Si los pesos de una máscara no suman 1.0, la imagen resultante
          se aclararía o se oscurecería drásticamente.

    ¿Qué devuelve?
        - mascara_normalizada (np.ndarray).
    """
    suma = np.sum(mascara)
    if suma != 0:
        return mascara / suma
    return mascara


# ---------------------------------------------------------------------------
# 3. PADDING (RELLENO DE BORDES)
# ---------------------------------------------------------------------------

def aplicar_padding_ceros(matriz_imagen, grosor_pad):
    """
    Agrega un borde de ceros alrededor de la matriz de la imagen.

    ¿Qué recibe?
        - matriz_imagen (np.ndarray): Matriz 2D (escala de grises) o 3D (color).
        - grosor_pad (int): Cantidad de filas y columnas de relleno (generalmente tamano_mascara // 2).

    ¿Qué operación realiza?
        - Si el grosor es 0, retorna la matriz tal como está.
        - Crea una matriz nueva más grande llena de ceros.
        - Copia la imagen original en el centro de la nueva matriz.

    ¿Por qué es necesaria?
        - Al aplicar una máscara de KxK sobre los píxeles del borde de la imagen,
          parte de la máscara quedaría fuera de la imagen. El padding evita errores
          de desbordamiento de índices y permite procesar los bordes uniformemente.

    ¿Qué devuelve?
        - imagen_con_padding (np.ndarray): Matriz de mayores dimensiones con ceros en el contorno.
    """
    if grosor_pad <= 0:
        return matriz_imagen.copy()

    # Si es imagen a color (3 dimensiones: alto, ancho, canales)
    if matriz_imagen.ndim == 3:
        alto, ancho, canales = matriz_imagen.shape
        nuevo_alto = alto + 2 * grosor_pad
        nuevo_ancho = ancho + 2 * grosor_pad

        # Creamos la matriz más grande llena de ceros
        imagen_con_padding = np.zeros((nuevo_alto, nuevo_ancho, canales), dtype=matriz_imagen.dtype)

        # Colocamos la imagen original en el centro
        imagen_con_padding[grosor_pad : grosor_pad + alto, grosor_pad : grosor_pad + ancho, :] = matriz_imagen

        return imagen_con_padding

    # Si es imagen en escala de grises (2 dimensiones: alto, ancho)
    else:
        alto, ancho = matriz_imagen.shape
        nuevo_alto = alto + 2 * grosor_pad
        nuevo_ancho = ancho + 2 * grosor_pad

        imagen_con_padding = np.zeros((nuevo_alto, nuevo_ancho), dtype=matriz_imagen.dtype)
        imagen_con_padding[grosor_pad : grosor_pad + alto, grosor_pad : grosor_pad + ancho] = matriz_imagen

        return imagen_con_padding


# ---------------------------------------------------------------------------
# 4. REDUCCIÓN CON SUAVIZADO (CONVOLUCIÓN + SALTO / STRIDE)
# ---------------------------------------------------------------------------

def reducir_con_suavizado(matriz_imagen, mascara, salto_fila=2, salto_columna=2):
    """
    Aplica suavizado con máscara y reduce las dimensiones de la imagen mediante saltos (strides).

    Sigue exactamente los 8 pasos solicitados:
        1. Obtener dimensiones de la matriz.
        2. Determinar el tamaño de la máscara y calcular el padding necesario.
        3. Aplicar padding con ceros a la imagen.
        4. Calcular el nuevo tamaño de salida y crear la matriz reducida.
        5. Recorrer los píxeles originales avanzando de salto_fila y salto_columna.
        6. Extraer la ventana y multiplicar por la máscara elemento a elemento.
        7. Sumar los productos de la multiplicación.
        8. Guardar el resultado en la matriz de salida.

    ¿Qué recibe?
        - matriz_imagen (np.ndarray): Matriz de entrada (2D o 3D).
        - mascara (np.ndarray): Matriz bidimensional de pesos (ej. 3x3 o 5x5).
        - salto_fila (int): Paso vertical entre muestras (stride vertical, ej. 2).
        - salto_columna (int): Paso horizontal entre muestras (stride horizontal, ej. 2).

    ¿Qué operación matemática realiza?
        - Convolución discreta centrada en cada punto de muestreo:
              Salida[f, c] = Suma( Ventana(i, j) * Mascara )
          donde los puntos (i, j) avanzan con pasos (salto_fila, salto_columna).

    ¿Por qué es necesaria?
        - Actúa como un filtro pasa-bajas anti-aliasing antes del submuestreo.
          Al eliminar las frecuencias espaciales altas, se evita que los detalles
          finos se transformen en patrones moiré o bordes dentados.

    ¿Qué devuelve?
        - matriz_reducida (np.ndarray): Nueva matriz con dimensiones reducidas y valores suavizados.

    ¿Qué ocurre si se modifican los saltos?
        - Si salto_fila = 2 y salto_columna = 2, la imagen se reduce a la mitad (50%).
        - Si salto_fila = 3 y salto_columna = 3, la imagen se reduce a la tercera parte (33%).
    """
    # 1. Obtener dimensiones de la imagen
    es_color = (matriz_imagen.ndim == 3)
    if es_color:
        alto_original, ancho_original, canales = matriz_imagen.shape
    else:
        alto_original, ancho_original = matriz_imagen.shape
        canales = 1

    # 2. Tamaño de la máscara y cálculo de padding
    tamano_mascara = mascara.shape[0]
    radio_pad = tamano_mascara // 2

    # 3. Aplicar padding de ceros
    imagen_con_padding = aplicar_padding_ceros(matriz_imagen, radio_pad)

    # 4. Calcular dimensiones de salida y crear matriz reducida
    # La cantidad de pasos es cuántos saltos caben a lo largo del alto y ancho
    alto_salida = math.ceil(alto_original / salto_fila)
    ancho_salida = math.ceil(ancho_original / salto_columna)

    if es_color:
        matriz_reducida = np.zeros((alto_salida, ancho_salida, canales), dtype=np.float64)
    else:
        matriz_reducida = np.zeros((alto_salida, ancho_salida), dtype=np.float64)

    # 5. Recorrer los píxeles con los saltos seleccionados
    fila_salida = 0
    for i in range(0, alto_original, salto_fila):
        col_salida = 0
        for j in range(0, ancho_original, salto_columna):

            # Extraemos la región correspondiente en la imagen con padding
            # El centro de la ventana corresponde exactamente al píxel original (i, j)
            if es_color:
                for c in range(canales):
                    # Ventana de tamaño (tamano_mascara, tamano_mascara)
                    ventana = imagen_con_padding[i : i + tamano_mascara, j : j + tamano_mascara, c]

                    # 6. Multiplicar los valores de la máscara por los píxeles correspondientes
                    producto = ventana * mascara

                    # 7. Sumar los productos
                    suma_productos = np.sum(producto)

                    # 8. Guardar el resultado en la nueva matriz de menor tamaño
                    matriz_reducida[fila_salida, col_salida, c] = suma_productos
            else:
                ventana = imagen_con_padding[i : i + tamano_mascara, j : j + tamano_mascara]
                producto = ventana * mascara
                suma_productos = np.sum(producto)
                matriz_reducida[fila_salida, col_salida] = suma_productos

            col_salida += 1
        fila_salida += 1

    return matriz_reducida


# ---------------------------------------------------------------------------
# 5. REDUCCIÓN DIRECTA SIN SUAVIZADO (SUBMUESTREO SIMPLE)
# ---------------------------------------------------------------------------

def reducir_directo_sin_suavizado(matriz_imagen, salto_fila=2, salto_columna=2):
    """
    Reduce la imagen tomando directamente un píxel cada cierto salto, sin aplicar ninguna máscara.

    ¿Qué recibe?
        - matriz_imagen (np.ndarray): Matriz original de la imagen.
        - salto_fila (int): Paso de filas (stride vertical).
        - salto_columna (int): Paso de columnas (stride horizontal).

    ¿Qué operación realiza?
        - Recorre la imagen con los saltos indicados y copia el píxel exacto
          de la coordenada (i, j) hacia la nueva matriz reducida.

    ¿Por qué es necesaria?
        - Sirve como método de CONTROL y COMPARACIÓN. Permite observar
          el aliasing (pixelado brusco, pérdida de continuidad) que ocurre
          cuando se reduce una imagen sin filtrar previamente las frecuencias altas.

    ¿Qué devuelve?
        - matriz_reducida (np.ndarray): Matriz reducida directamente.
    """
    es_color = (matriz_imagen.ndim == 3)
    if es_color:
        alto_original, ancho_original, canales = matriz_imagen.shape
    else:
        alto_original, ancho_original = matriz_imagen.shape
        canales = 1

    alto_salida = math.ceil(alto_original / salto_fila)
    ancho_salida = math.ceil(ancho_original / salto_columna)

    if es_color:
        matriz_reducida = np.zeros((alto_salida, ancho_salida, canales), dtype=matriz_imagen.dtype)
    else:
        matriz_reducida = np.zeros((alto_salida, ancho_salida), dtype=matriz_imagen.dtype)

    fila_salida = 0
    for i in range(0, alto_original, salto_fila):
        col_salida = 0
        for j in range(0, ancho_original, salto_columna):
            if es_color:
                matriz_reducida[fila_salida, col_salida, :] = matriz_imagen[i, j, :]
            else:
                matriz_reducida[fila_salida, col_salida] = matriz_imagen[i, j]
            col_salida += 1
        fila_salida += 1

    return matriz_reducida


# ---------------------------------------------------------------------------
# 6. MÉTRICA DE COMPARACIÓN (ERROR CUADRÁTICO MEDIO - MSE)
# ---------------------------------------------------------------------------

def calcular_diferencia_absoluta(matriz_con_suavizado, matriz_sin_suavizado):
    """
    Calcula la diferencia absoluta promedio entre la imagen reducida con suavizado
    y la imagen reducida sin suavizado.

    ¿Qué recibe?
        - matriz_con_suavizado (np.ndarray)
        - matriz_sin_suavizado (np.ndarray)

    ¿Qué operación matemática realiza?
        - Diferencia = Promedio( | con_suavizado - sin_suavizado | )

    ¿Por qué es necesaria?
        - Cuantifica numéricamente cuánto difiere la imagen suavizada de la
          muestra directa (demuestra el impacto del filtro en los números).

    ¿Qué devuelve?
        - diferencia_promedio (float): Magnitud del cambio en los niveles de gris/color.
    """
    diferencia = np.abs(matriz_con_suavizado.astype(np.float64) - matriz_sin_suavizado.astype(np.float64))
    return float(np.mean(diferencia))
