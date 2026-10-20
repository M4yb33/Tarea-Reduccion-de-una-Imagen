# Guía breve para la defensa: convolución y reducción manual

**Tema:** Procesamiento digital de imágenes: suavizado mediante máscaras y reducción controlada por saltos.

## Pregunta 1. ¿Qué problema resuelve el software?

Permite reducir la resolución de una fotografía RGB sin utilizar funciones automáticas de reducción. Para mitigar aliasing, utiliza una máscara de suavizado programada manualmente antes de escoger los píxeles de salida.

## Pregunta 2. ¿Qué es una máscara?

Es una matriz de pesos que se coloca sobre una región pequeña de la fotografía. Con 3×3 se consideran nueve posiciones vecinas. El tamaño es impar para que exista un centro exacto.

## Pregunta 3. ¿Qué significa convolucionar?

Multiplicar cada intensidad de la vecindad por el peso correspondiente, sumar los productos y asignar el resultado a una posición de salida. Una convolución matemática invierte la máscara antes de multiplicar; en las máscaras simétricas da lo mismo.

## Pregunta 4. ¿Cómo se forma una máscara gaussiana 3×3?

A partir del vector `[1, 2, 1]` y su transpuesto:

```text
[1]                 [1 2 1]
[2]    × [1 2 1] =  [2 4 2]
[1]                 [1 2 1]
```

La suma de pesos es 16; se divide toda la máscara entre 16:

```text
1/16 * [1 2 1]
       [2 4 2]
       [1 2 1]
```

Para tamaños 5×5 y 7×7, el programa calcula pesos gaussianos mediante una fórmula exponencial y su producto exterior.

## Pregunta 5. ¿Por qué sumar o promediar vecinos reduce ruido?

Porque las diferencias bruscas aisladas se atenúan al combinar intensidades cercanas. A la vez se suavizan bordes y detalles; no existe un filtro que conserve intactos todos los detalles y elimine cualquier ruido.

## Pregunta 6. ¿Qué es el padding?

Es un borde artificial añadido para que el centro de la máscara pueda situarse también sobre un píxel de la esquina. Para una máscara 3×3 se añade un borde de grosor 1; para 5×5, grosor 2. El software ofrece ceros y reflejo. El padding de ceros puede oscurecer esquinas.

## Pregunta 7. ¿Qué significa salto de 2×2?

Se calcula una muestra de salida cada dos filas y cada dos columnas de la fotografía de entrada. Así, aproximadamente se reduce la anchura a la mitad y la altura a la mitad: se conserva cerca de un cuarto de los píxeles, es decir, una reducción de ~75 %.

## Pregunta 8. ¿El tamaño de la máscara es lo mismo que el salto?

No. La máscara define **cuántos vecinos contribuyen** a un píxel; el salto define **qué tan separados están** los centros de los píxeles de salida. Por eso pueden usarse, por ejemplo, máscara 5×5 y salto 2×3.

## Pregunta 9. ¿Qué diferencia existe entre reducción directa y filtrada?

La reducción directa toma valores puntuales, sin considerar vecinos. La filtrada combina vecinos antes del muestreo y puede disminuir aliasing. Ambos resultados tienen las mismas dimensiones cuando usan idénticos saltos.

## Pregunta 10. ¿Cuál es el resultado real sobre el paisaje?

La fotografía de 626×417 contiene 261.042 píxeles. Con salto 2×2 pasa a 313×209 y contiene 65.417 píxeles. La cantidad disminuye un 74,94 %. La imagen original y la imagen suavizada conservan sus dimensiones antes de la reducción.

## Pregunta 11. ¿Dónde se encuentran las fórmulas en el código?

En `procesamiento.py`:

1. `crear_mascara()`: crea y normaliza pesos; gaussiana a partir del producto exterior de dos vectores.
2. `aplicar_convolucion()`: añade padding; define el tamaño final; recorre las filas y columnas de la máscara; multiplica regiones y pesos y suma.
3. `reducir_sin_suavizado()`: selecciona píxeles de referencia sin filtrado.
4. `calcular_datos()`: calcula cantidad de píxeles y porcentaje de reducción.
5. `explicar_pixel()`: permite mostrar cada multiplicación numérica en una vecindad real.

En `main.py`, `procesar()` toma los valores de la interfaz y llama a estas funciones.

## Pregunta 12. ¿Por qué se usa NumPy si la convolución debía ser manual?

NumPy se usa para representar y operar matrices. Los dos bucles `for` de `aplicar_convolucion` implementan la suma de los productos por cada peso de la máscara. La biblioteca no decide la máscara, el padding, la multiplicación, la suma ni los saltos. Operar varias posiciones a la vez acelera la interfaz sin utilizar una función ya hecha de convolución o reducción.

## Pregunta 13. ¿Se utiliza `resize()`?

**Solo para presentar una imagen dentro de un panel Tkinter**, después de haber calculado la matriz. La reducción matemática se realiza sin `resize()`. El PNG exportado es exactamente la matriz reducida, con sus dimensiones reales.

## Pregunta 14. ¿Qué significa la variación espacial?

Es un promedio de diferencias absolutas entre píxeles vecinos. Si disminuye tras suavizar, la foto tiene variaciones locales más pequeñas. **No es una calificación de calidad visual**: incluso un desenfoque exagerado puede producir una variación muy baja.

## Ejercicio manual que debe dominar cada integrante

Una vecindad de intensidad de un canal:

```text
100 120 140
 90 110 130
 80 100 120
```

Con la máscara `[[1,2,1],[2,4,2],[1,2,1]] / 16`:

```text
100(1) + 120(2) + 140(1)
+ 90(2) + 110(4) + 130(2)
+ 80(1) + 100(2) + 120(1)
= 1760
```

Por tanto `1760 / 16 = 110`. El valor procesado del píxel central es **110**.

## Guion de exposición para dos estudiantes

**Integrante 1 (fundamentos y parámetros):** presentar el objetivo; explicar matriz RGB, máscara impar, media gaussiana, padding de ceros; demostrar la matriz 3×3 y el cálculo manual de 110; modificar el tipo y tamaño de máscara.

**Integrante 2 (algoritmo y resultados):** explicar `aplicar_convolucion`; marcar los bucles de la máscara, la multiplicación y la suma; mostrar los saltos horizontal/vertical y comparar las cuatro vistas; comprobar 626×417 → 313×209 y guardar el PNG.

**Ambos deben poder intercambiar funciones.** En la demostración se recomienda fijar primero Gaussiana 3×3, salto 2×2 y padding de ceros. Si el profesor pregunta por bordes, cambiar a reflejo y observar la diferencia. Para comparar aliasing, observar reducción directa y con máscara en texturas finas.

## Resultado verificable

Ejecutar `python pruebas.py` antes de la defensa. Las comprobaciones incluyen promedios, gaussiana, padding, saltos, máscara asimétrica, consistencia del cálculo por píxel, dimensiones, reproducibilidad, guardado PNG y filtrado de un patrón ajedrezado.
