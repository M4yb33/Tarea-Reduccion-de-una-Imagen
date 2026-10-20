# Guía para la Defensa: Taller de Suavizado y Reducción de Imágenes

**Asignatura:** Inteligencia Artificial (Segundo Parcial)  
**Institución:** Universidad Técnica de Ambato  
**Integrantes:** 2 estudiantes  

---

## 1. Preguntas Teóricas y Matemáticas Fundamentales

### Pregunta 1. ¿Qué problema resuelve este software?
Permite reducir la resolución espacial de una fotografía digital RGB implementando manualmente el algoritmo de convolución y saltos (*strides*), sin recurrir a bibliotecas prediseñadas como `cv2.resize()` o `PIL.Image.resize()`. Aplica un suavizado previo con máscara para mitigar el efecto de **aliasing** (distorsión por submuestreo).

### Pregunta 2. ¿Qué es una máscara de convolución y por qué tiene tamaño impar?
Es una matriz pequeña de coeficientes (pesos) que se superpone sobre una vecindad de píxeles. Tiene tamaño impar ($3\times3, 5\times5, 7\times7$) para poseer un **centro geométrico exacto**, el cual coincide con la coordenada del píxel a procesar.

### Pregunta 3. ¿Cómo se normaliza una máscara y por qué es obligatorio?
Se suman todos los valores de la matriz y se divide cada coeficiente entre dicha suma:
$$M_{\text{norm}} = \frac{M}{\sum_{i,j} M_{i,j}}$$
Es obligatorio para que la suma total de los pesos sea igual a **1.0**. Si la suma fuera mayor a 1, la imagen se sobreexpondría (se volvería blanca); si fuera menor a 1, la imagen se oscurecería. Con suma 1.0, el brillo medio de la fotografía se conserva.

### Pregunta 4. ¿Cómo se construye la máscara gaussiana 3×3?
A partir del vector unidimensional discreto $[1, 2, 1]$ y su producto exterior con el vector transpuesto:
$$\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
La suma de los pesos es $1+2+1+2+4+2+1+2+1 = 16$. Por tanto, la máscara gaussiana normalizada 3×3 es:
$$M_{\text{gauss}} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

### Pregunta 5. ¿Qué es el padding de ceros y qué efecto produce?
Para que el centro de la máscara pueda posicionarse sobre los píxeles de los bordes y esquinas de la imagen, se añade un contorno artificial de ceros de grosor:
$$p = \frac{K - 1}{2}$$
- Para máscara 3×3: 1 píxel de ceros por lado.
- Para máscara 5×5: 2 píxeles de ceros por lado.
- Para máscara 7×7: 3 píxeles de ceros por lado.

**Efecto en los bordes:** En las esquinas y bordes extremos, parte de los vecinos son ceros artificiales. Por tanto, la suma ponderada en el borde disminuye, lo que produce un ligero oscurecimiento en los márgenes de la imagen. Esto es matemáticamente normal y propio del padding de ceros.

### Pregunta 6. ¿Cuál es la diferencia entre convolución y correlación?
- La **correlación cruzada** multiplica la vecindad directamente por la máscara tal como está orientada.
- La **convolución matemática** primero invierte la máscara en ambos ejes (rotación de 180°: `W[::-1, ::-1]`) antes de multiplicar y sumar.
- En máscaras simétricas (Promedio y Gaussiana), invertir la máscara da exactamente el mismo resultado.
- En máscaras personalizadas asimétricas, el algoritmo implementa convolución verdadera invirtiendo los índices.

### Pregunta 7. ¿Qué significa un salto (stride) de 2×2?
Indica que el centro de la máscara avanza de 2 en 2 filas y de 2 en 2 columnas sobre la imagen original. De este modo, se calcula una muestra por cada cuatro píxeles originales, reduciendo la anchura y la altura aproximadamente a la mitad.
Las dimensiones resultantes son:
$$H_{\text{salida}} = \left\lfloor\frac{H - 1}{s_v}\right\rfloor + 1, \qquad W_{\text{salida}} = \left\lfloor\frac{W - 1}{s_h}\right\rfloor + 1$$
Para la fotografía de `626 × 417` con salto `2×2`:
$$H_{\text{salida}} = \left\lfloor\frac{417 - 1}{2}\right\rfloor + 1 = 208 + 1 = 209$$
$$W_{\text{salida}} = \left\lfloor\frac{626 - 1}{2}\right\rfloor + 1 = 312 + 1 = 313$$
Resolución reducida: **313 × 209 píxeles**.

### Pregunta 8. ¿Qué diferencia existe entre reducción directa y reducción con suavizado?
- **Reducción directa (sin filtro):** Toma los valores puntuales de la imagen original en las posiciones de salto (`imagen[::sy, ::sx]`). Es propensa a **aliasing**, produciendo bordes dentados y pérdida de patrones finos.
- **Reducción con suavizado (convolución):** Antes de muestrear, promedia las frecuencias altas de la vecindad con la máscara de suavizado, actuando como un **filtro paso-bajo anti-aliasing**.
- Ambas reducciones producen imágenes con **las mismas dimensiones exactas**.

### Pregunta 9. ¿Se utiliza `resize()` en el procesamiento?
**No para el cálculo matemático.** La reducción matemática la realizan los bucles de convolución y saltos en `procesamiento.py`. La biblioteca `Pillow (resize)` se utiliza **únicamente para adaptar la visualización de la matriz resultante al tamaño del panel de Tkinter**, garantizando que la fotografía se aprecie de forma nítida y cómoda sin distorsionar la matriz numérica. El archivo PNG guardado contiene exactamente los píxeles reducidos.

### Pregunta 10. ¿Por qué un píxel seleccionado en la foto original puede no formar parte de la reducción?
Con origen en cero y saltos $s_v$ y $s_h$, la reducción solo conserva las posiciones que cumplen:
$$\text{fila} \bmod s_v = 0 \quad \text{y} \quad \text{columna} \bmod s_h = 0$$
Por ejemplo, con salto 2×2, solo se muestrean las filas y columnas pares. Si el usuario hace clic en el píxel `(209, 313)`, esa posición no es par; por ende, el cálculo de convolución paso a paso es completamente válido como demostración matemática local, pero ese píxel no forma parte de la matriz reducida final. La interfaz informa esto explícitamente.

### Pregunta 11. ¿Cómo se garantiza que el clic en pantalla seleccione exactamente el píxel deseado sin desfasajes?
El software utiliza la función unificada `calcular_geometria_panel()` tanto para dibujar la imagen centrada con su margen interior de 8 píxeles en el lienzo como para convertir las coordenadas `(x, y)` del clic del ratón. Al compartir exactamente el mismo factor de escalado y los mismos desplazamientos de origen `(ox, oy)`, se elimina cualquier error o desfase de columnas y filas (por ejemplo, en un panel de 500×400, un clic en el margen izquierdo selecciona exactamente la columna 0).

### Pregunta 12. ¿Por qué el cálculo paso a paso y la convolución general coinciden al 100% en todos los píxeles?
Porque internamente las operaciones matemáticas se ejecutan con precisión flotante completa de 64 bits (`np.float64`) y la máscara gaussiana 3×3 se genera de forma exacta mediante el producto exterior del vector $[1, 2, 1]$. La interfaz muestra los pesos con 4 decimales para comodidad visual del usuario, pero el cálculo matemático de los productos y la suma utiliza los pesos exactos con `np.rint()`, garantizando idéntico resultado píxel a píxel (por ejemplo, en la fila 106, columna 576, canal verde de `paisaje.png`, ambos producen exactamente 76).

### Pregunta 13. ¿Qué ocurre si se ingresa una máscara personalizada inválida (como nueve ceros o valores negativos)?
El sistema aplica una validación estricta: rechaza la entrada y despliega un mensaje explicativo indicando el motivo del error. En ningún caso sustituye silenciosamente los valores por un filtro de promedio ni altera los números ingresados por el usuario, permitiéndole corregir la matriz directamente.

---

## 2. Ejercicio Manual de Convolución (Dominio Obligatorio)

Dada la vecindad de intensidades $3\times3$ de un canal:
$$\begin{bmatrix} 100 & 120 & 140 \\ 90 & 110 & 130 \\ 80 & 100 & 120 \end{bmatrix}$$
Con la máscara Gaussiana $3\times3$ dividida entre 16:
$$\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} / 16$$
Cálculo de productos elemento a elemento:
$$\begin{aligned}
S &= 100(1) + 120(2) + 140(1) \\
  &\quad + 90(2) + 110(4) + 130(2) \\
  &\quad + 80(1) + 100(2) + 120(1) \\
S &= 100 + 240 + 140 + 180 + 440 + 260 + 80 + 200 + 120 = 1760
\end{aligned}$$
Dividiendo entre la suma de pesos:
$$\text{Píxel final} = \frac{1760}{16} = \mathbf{110}$$
El valor del píxel central suavizado es **110**. Este ejercicio se comprueba de forma automática en la **Prueba 9** de `pruebas.py`.

---

## 3. Guion de Exposición para los Dos Estudiantes

Ambos integrantes dominan todo el proyecto, pero se dividen la exposición para una presentación ágil y estructurada:

### Integrante 1: Fundamentos Matemáticos, Parámetros y Selección Interactiva
1. **Presentación:** Saludar al docente y explicar el objetivo: reducción manual de imágenes RGB mitgando aliasing mediante convolución y saltos.
2. **Carga y Estructura:** Mostrar la imagen cargada (`paisaje.png`, `626 × 417 px`, 261.042 píxeles, 3 canales RGB `uint8`).
3. **Máscaras de Suavizado:**
   - Seleccionar **Gaussiana 3×3**: explicar la matriz $[1, 2, 1] \times [1, 2, 1] / 16$, la suma 1.0 y por qué las casillas están bloqueadas como solo lectura.
   - Seleccionar **Promedio 3×3**: explicar que cada celda vale $1/9$.
   - Seleccionar **Personalizada**: demostrar que las casillas se habilitan para edición. Probar los presets rápidos (`1s`, `1-2-1`, `Centro 5`) y modificar un número con validación estricta (no admite letras ni negativos).
4. **Padding de Ceros:** Explicar que se añade un marco de $p = (K-1)/2$ ceros para procesar los bordes, justificando el ligero oscurecimiento en márgenes extremos.
5. **Selección por Clic:** Hacer clic sobre la fotografía original y mostrar cómo se marca discretamente con una retícula roja, mostrando en la barra lateral: `Píxel seleccionado: fila X, columna Y`.

### Integrante 2: Algoritmo de Convolución, Reducción, Saltos y Cálculo Paso a Paso
1. **Convolución con Saltos:** Explicar los bucles `for` en `aplicar_convolucion` de `procesamiento.py`: recorre las filas y columnas de la máscara, invierte el kernel, multiplica la región con saltos y suma los productos.
2. **Comparación de las Cuatro Vistas:**
   - **Original** (`626 × 417`) vs **Suavizada** (`626 × 417`).
   - **Reducción Directa** (`313 × 209`) vs **Reducción con Suavizado** (`313 × 209`).
   - Destacar cómo el suavizado previo mitiga el aliasing en los detalles finos y texturas del paisaje.
3. **Resumen de Resultados:** Mostrar la tarjeta lateral con la reducción del `74,94 %` (de 261.042 a 65.417 píxeles) y el tiempo de cómputo en milisegundos.
4. **Ventana de Cálculo Paso a Paso:**
   - Abrir **Ver cálculo paso a paso — Convolución**.
   - Mostrar la descomposición: $\text{Píxeles } I \times \text{Máscara } W = \text{Productos } P \rightarrow \text{Suma } S \rightarrow \text{Píxel de salida}$.
   - Explicar el indicador de submuestreo: si la posición cumple $f \bmod s_v == 0$ y $c \bmod s_h == 0$.
   - Cambiar los canales (Rojo, Verde, Azul).
   - Realizar el ejercicio manual de intensidades ($100, 110, 120 \rightarrow 110$).
5. **Botón Restaurar y Guardar PNG:**
   - Pulsar **Restaurar**: demostrar que regresa a la configuración inicial (Gaussiana 3×3, salto 2×2).
   - Pulsar **Guardar PNG**: verificar que el archivo generado tiene exactamente las dimensiones reducidas reales (313×209).

---

## 4. Verificación de las 18 Pruebas Automatizadas

Antes de la defensa, ejecutar en la terminal:
```bash
python pruebas.py
```
Todas las pruebas finalizarán con código de salida 0:
- **Prueba 1:** Padding de ceros en esquinas y bordes.
- **Prueba 2:** Máscara de promedio con pesos uniformes ($1/9$).
- **Prueba 3:** Máscara gaussiana normalizada a 1.0.
- **Prueba 4:** Edición deshabilitada en Promedio y Gaussiana (`readonly`).
- **Prueba 5:** Edición habilitada en Personalizada (`normal`).
- **Prueba 6:** Validación de valores numéricos personalizados.
- **Prueba 7:** Conversión precisa de clic en pantalla a fila y columna.
- **Prueba 8:** Clic fuera de la fotografía ignorado sin errores.
- **Prueba 9:** Ejercicio matemático central coincidente con convolución (suma/16 = 110).
- **Prueba 10:** Identificación de posiciones de submuestreo para salto 2×2.
- **Prueba 11:** Salto 2×2 sobre 626×417 genera resolución 313×209.
- **Prueba 12:** Saltos independientes horizontal y vertical.
- **Prueba 13:** Igualdad de dimensiones entre reducción directa y suavizada.
- **Prueba 14:** Conservación de los 3 canales RGB independientes en `uint8`.
- **Prueba 15:** Botón Restaurar restableciendo la configuración inicial.
- **Prueba 16:** Exportación PNG con matriz reducida exacta.
- **Prueba 17:** Reproducibilidad determinista de resultados.
- **Prueba 18:** Visualización ampliada en pantalla sin distorsión de los datos reales exportados.
