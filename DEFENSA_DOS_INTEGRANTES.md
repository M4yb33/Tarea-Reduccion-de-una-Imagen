# Guía para la Defensa: Taller de Suavizado y Reducción de Imágenes

**Asignatura:** Inteligencia Artificial (Segundo Parcial)  
**Institución:** Universidad Técnica de Ambato  
**Integrantes:** 2 estudiantes  

---

## 1. Preguntas Teóricas y Matemáticas Fundamentales

### Pregunta 1. ¿Qué problema resuelve este software?
Permite reducir la resolución espacial de una fotografía digital RGB implementando manualmente el algoritmo de convolución y saltos (*strides*), sin recurrir a bibliotecas prediseñadas como `cv2.resize()` o `PIL.Image.resize()`. Aplica un suavizado previo con máscara para mitigar el efecto de **aliasing** (distorsión por submuestreo).

### Pregunta 2. ¿Por qué se pasó de cuatro vistas a TRES vistas en la interfaz?
La versión final simplificada elimina la vista independiente «IMAGEN SUAVIZADA» a resolución completa porque:
1. **Enfoque directo de comparación:** El objetivo pedagógico central es comparar la **reducción directa** (submuestreo crudo sin filtro) con la **reducción con suavizado** (convolución previa + saltos).
2. **Eficiencia computacional:** Al calcular la reducción con saltos directamente en `aplicar_convolucion(imagen, mascara, sv, sh)`, se evalúan únicamente las vecindades que generan los píxeles de salida, evitando calcular innecesariamente una imagen suavizada completa de millones de operaciones con salto 1×1.
3. **Equilibrio visual:** Dispone la **imagen original grande a la izquierda** y las **dos reducciones apiladas a la derecha**, permitiendo inspeccionar cómodamente los efectos visuales sin distorsión de relación de aspecto.

### Pregunta 3. ¿Qué es una máscara de convolución y por qué tiene tamaño impar?
Es una matriz pequeña de coeficientes (pesos) que se superpone sobre una vecindad de píxeles. Tiene tamaño impar ($3\times3, 5\times5, 7\times7$) para poseer un **centro geométrico exacto**, el cual coincide con la coordenada del píxel a procesar.

### Pregunta 4. ¿Cómo se normaliza una máscara y por qué es obligatorio?
Se divide cada coeficiente entre la suma de todos los pesos de la matriz:
$$M_{\text{norm}} = \frac{M}{\sum_{i,j} M_{i,j}}$$
Es obligatorio para que la suma total de los pesos sea igual a **1.0**. Si la suma fuera mayor a 1, la imagen se sobreexpondría; si fuera menor a 1, se oscurecería. Con suma 1.0, el brillo medio de la fotografía se conserva.

### Pregunta 5. ¿Qué sucedía con los pesos personalizados en el cálculo paso a paso y cómo se corrigió?
En versiones previas, la edición manual de pesos personalizados multiplicaba las intensidades por los pesos crudos sin normalizar (por ejemplo, en una región uniforme de 100 con pesos que sumaban 1,25, el resultado arrojaba $100 \times 1,25 = 125$).  
**Corrección implementada:** Ahora la ventana paso a paso invoca la función central `crear_mascara('Personalizada', tam, pesos)`, normalizando la matriz a `float64` antes de multiplicar. Para intensidades uniformes de 100, la suma de productos normalizada produce exactamente **100**, coincidiendo con la operación general de `aplicar_convolucion()`.

### Pregunta 6. ¿Cómo se construye la máscara gaussiana 3×3?
A partir del vector discreto $[1, 2, 1]$ y su producto exterior:
$$\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
Dividiendo entre la suma (16), se obtiene la máscara normalizada:
$$M_{\text{gauss}} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

### Pregunta 7. ¿Qué es el padding de ceros y qué efecto produce?
Para que el centro de la máscara pueda posicionarse sobre los bordes y esquinas de la imagen, se añade un contorno artificial de ceros de grosor $p = (K - 1) / 2$:
- Para máscara 3×3: $p = 1$.
- Para máscara 5×5: $p = 2$.
- Para máscara 7×7: $p = 3$.  
*Efecto:* En los márgenes extremos, los ceros artificiales atenúan ligeramente la suma ponderada, produciendo un oscurecimiento leve y esperado matemáticamente.

### Pregunta 8. ¿Cuál es la diferencia entre convolución y correlación?
- La **correlación cruzada** multiplica la vecindad directamente por la máscara sin invertir.
- La **convolución matemática** invierte previamente la máscara en ambos ejes ($W[::-1, ::-1]$).
- En máscaras simétricas (Promedio y Gaussiana), invertir da el mismo resultado.
- En máscaras asimétricas, el algoritmo invierte los coeficientes tanto en el procesamiento global como en el cálculo paso a paso, garantizando coincidencia idéntica sin dobles inversiones erróneas.

### Pregunta 9. ¿Qué significa un salto (stride) de 2×2?
Indica que el centro de la máscara avanza de 2 en 2 filas y de 2 en 2 columnas sobre la imagen original:
$$H_{\text{salida}} = \left\lfloor\frac{H - 1}{s_v}\right\rfloor + 1, \qquad W_{\text{salida}} = \left\lfloor\frac{W - 1}{s_h}\right\rfloor + 1$$
Para la fotografía de `626 × 417` con salto `2×2`:
$$H_{\text{salida}} = \left\lfloor\frac{417 - 1}{2}\right\rfloor + 1 = 209, \qquad W_{\text{salida}} = \left\lfloor\frac{626 - 1}{2}\right\rfloor + 1 = 313$$
Resolución reducida: **313 × 209 píxeles** (reducción del 74,94 % de píxeles).

### Pregunta 10. ¿Por qué un píxel seleccionado en la foto original puede no formar parte de la reducción?
La reducción con origen $(0, 0)$ solo conserva las posiciones que cumplen:
$$\text{fila} \bmod s_v = 0 \quad \text{y} \quad \text{columna} \bmod s_h = 0$$
Con salto 2×2, solo se conservan filas y columnas pares. Si el usuario hace clic en el píxel `(209, 313)`, la ventana didáctica calcula la convolución demostrativa válida localmente, pero el indicador señala en naranja que dicha posición no forma parte de la matriz reducida final.

### Pregunta 11. ¿Cómo se garantiza que el clic en pantalla seleccione el píxel deseado sin desfasajes?
El software utiliza la función unificada `calcular_geometria_panel()` y `convertir_clic_a_pixel()` empleando las dimensiones enteras efectivamente dibujadas (`vw, vh`) y los desplazamientos de centrado (`ox, oy`):
$$\text{columna} = \left\lfloor\frac{(\text{clic}_x - ox) \times W_{\text{orig}}}{vw}\right\rfloor, \qquad \text{fila} = \left\lfloor\frac{(\text{clic}_y - oy) \times H_{\text{orig}}}{vh}\right\rfloor$$
Al utilizar las proporciones de visualización real, un clic en el borde izquierdo mapea a la columna 0, y los clics fuera del área visible se descartan sin alterar la selección previa.

### Pregunta 12. ¿Cómo se garantizó que la matriz 7×7 quepa completa en el panel lateral sin recortes?
Se configuraron dimensiones dinámicas en [`actualizar_mascara()`](file:///f:/UTA/OCTAVO/INTELIGENCIA%20ARTIFICIAL/2p/deberes/Tarea-Reduccion-de-una-Imagen/main.py#L216) según el tamaño del kernel:
- **3×3:** ancho 6, fuente `Consolas 9`, padding 2 px.
- **5×5:** ancho 5, fuente `Consolas 8`, padding 1 px.
- **7×7:** ancho 4 (solo lectura, 3 decimales) y 3 (editable), fuente `Consolas 7/8`, padding 1 px.
Con este ajuste, la matriz 7×7 requiere un ancho de tan solo 182 px, alojándose holgadamente en el contenedor lateral de 308 px (`sb` de 330 px). Todas las 49 casillas son 100% visibles y legibles, incluso con escalados altos de DPI en Windows.

### Pregunta 13. ¿Cómo se sincroniza la ventana de cálculo paso a paso al cambiar la máscara o la imagen?
Si la ventana de cálculo está abierta y el usuario cambia el tamaño o tipo de máscara en la ventana principal, el sistema detecta el cambio, cierra de forma segura la ventana anterior limpiando callbacks, y la reabre automáticamente con el nuevo tamaño (por ejemplo de 3×3 a 7×7 con 49 celdas) preservando el píxel seleccionado.
- **Cambio de saltos:** Se realiza una sincronización ligera del indicador de muestreo (`callback_actualizar_salto`) sin reconstruir la ventana.
- **Cambio de imagen:** Se reinicializan límites y vecindad con el centro de la nueva fotografía.
- **Aplicación desde la ventana:** Una bandera de exclusión (`_aplicando_desde_calculo`) evita que la ventana se destruya mientras su botón todavía procesa el evento.

---

## 2. Ejercicio Manual de Convolución (Dominio Obligatorio)

Dada la vecindad de intensidades $3\times3$ de un canal:
$$\begin{bmatrix} 100 & 120 & 140 \\ 90 & 110 & 130 \\ 80 & 100 & 120 \end{bmatrix}$$
Con la máscara Gaussiana $3\times3$ dividida entre 16:
$$\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$
Cálculo de productos elemento a elemento:
$$\begin{aligned}
S &= 100(1) + 120(2) + 140(1) \\
  &\quad + 90(2) + 110(4) + 130(2) \\
  &\quad + 80(1) + 100(2) + 120(1) \\
S &= 100 + 240 + 140 + 180 + 440 + 260 + 80 + 200 + 120 = 1760
\end{aligned}$$
Dividiendo entre la suma de pesos:
$$\text{Píxel final} = \frac{1760}{16} = \mathbf{110}$$
El valor del píxel central suavizado es **110**. Este cálculo se comprueba automáticamente en la **Prueba 9** de `pruebas.py`.

---

## 3. Guion de Exposición para los Dos Estudiantes

Ambos integrantes dominan todo el proyecto, dividiéndose la presentación de forma fluida:

### Integrante 1: Fundamentos Matemáticos, Parámetros y Selección Interactiva
1. **Presentación:** Saludar al docente y explicar el propósito: reducir la resolución de imágenes RGB mitigando aliasing mediante convolución manual y saltos.
2. **Carga y Estructura:** Mostrar la fotografía cargada (`paisaje.png`, `626 × 417 px`, 261.042 píxeles, tensor RGB `uint8`).
3. **Máscaras de Suavizado:**
   - **Gaussiana 3×3:** Explicar el kernel $[1, 2, 1] \times [1, 2, 1] / 16$, la suma 1.0 y las casillas bloqueadas en solo lectura.
   - **Promedio 3×3:** Explicar los pesos idénticos $1/9$.
   - **Personalizada:** Demostrar que las casillas se habilitan para edición. Probar los presets (`1s`, `1-2-1`, `Centro 5`) y mostrar la validación estricta que rechaza valores vacíos, negativos o ceros sin sustitución silenciosa.
4. **Padding de Ceros:** Explicar el marco de ceros $p = (K-1)/2$ para procesar los bordes, justificando el oscurecimiento marginal.
5. **Selección por Clic:** Hacer clic en el centro y en los bordes de la foto original; mostrar cómo la retícula roja se posiciona sin desfase y se reporta en la barra lateral.

### Integrante 2: Convolución con Saltos, Comparación de 3 Vistas y Cálculo Paso a Paso
1. **Convolución con Saltos:** Explicar los bucles `for` en `aplicar_convolucion` de `procesamiento.py`: recorre las filas y columnas de la máscara, invierte el kernel, multiplica con saltos y suma los productos.
2. **Las Tres Vistas de la Interfaz:**
   - **Original** (`626 × 417`) a la izquierda.
   - **Reducción Directa** (`313 × 209`) arriba a la derecha: mostrar el aliasing en detalles y bordes.
   - **Reducción con Suavizado** (`313 × 209`) abajo a la derecha: destacar cómo la convolución previa suaviza transiciones bruscas antes de muestrear.
   - Explicar por qué se eliminó la cuarta imagen suavizada completa (optimización de cómputo y comparación focalizada).
3. **Tarjeta de Resultados:** Conteo de píxeles ($261.042 \rightarrow 65.417$), porcentaje de reducción ($74,94 \%$) y tiempo de cálculo en ms.
4. **Ventana de Cálculo Paso a Paso:**
   - Abrir **Ver cálculo paso a paso — Convolución**.
   - Mostrar $I \times W_{\text{conv}} = P \rightarrow \text{Suma} \rightarrow \text{Píxel final}$.
   - Demostrar que pesos personalizados que suman 1,25 se normalizan correctamente y dan 100 en regiones uniformes de 100.
   - Mostrar accesibilidad completa en 3×3, 5×5 y 7×7 con el desplazamiento horizontal.
   - Probar el indicador de submuestreo con el salto 2×2.
5. **Restaurar y Guardar PNG:**
   - Demostrar que **Restaurar** restablece Gaussiana 3×3 y salto 2×2.
   - Demostrar que **Guardar PNG** exporta exactamente la matriz reducida real de 313×209 píxeles.

---

## 4. Resumen de Verificación y Pruebas Automatizadas

Antes de la defensa, ejecutar en la terminal:
```bash
python pruebas.py
python prueba_interfaz.py
```
- `pruebas.py`: ejecuta con éxito las **18 pruebas matemáticas obligatorias** más las pruebas de equivalencia con salida 0.
- `prueba_interfaz.py`: ejecuta con éxito las **19 pruebas de interfaz y sincronización**, validando las 3 vistas, visibilidad completa de matriz 7×7 sin recorte, Casos A al G de sincronización de cálculo paso a paso, normalización, clics en bordes y exportación.
