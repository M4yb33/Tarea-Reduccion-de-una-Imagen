# Taller de Inteligencia Artificial: suavizado y reducción por convolución

**Proyecto didáctico, funcional y fácil de explicar para dos estudiantes.**

El programa permite cargar fotografías RGB, elegir una máscara impar (3×3, 5×5 o 7×7), introducir valores propios y seleccionar cuántas filas y columnas avanzar. Se obtienen cuatro vistas: imagen original, imagen suavizada, reducción directa sin filtro y reducción con suavizado. Las imágenes procesadas se crean con código propio; ninguna función automática realiza el filtrado o la reducción.

## 1. Instalación y ejecución

Se requiere **Python 3.10 o posterior**.

Desde la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
python main.py
```

El programa carga automáticamente `imagenes/paisaje.png` si está disponible. Se puede seleccionar cualquier otra fotografía JPG/PNG usando **Cargar imagen**.

Para ejecutar comprobaciones:

```bash
python pruebas.py
```

En Linux sin escritorio, las pruebas de GUI requieren `xvfb-run`:

```bash
xvfb-run -a python prueba_interfaz.py
```

En Windows, con el escritorio iniciado, basta `python prueba_interfaz.py`.

## 2. Qué hace cada control

| Control | Función |
|---|---|
| Máscara **Gaussiana** | Da mayor peso a los píxeles cercanos al centro para suavizar |
| Máscara **Promedio** | Todos los píxeles de la ventana tienen igual peso |
| Máscara **Personalizada** | El usuario escribe una matriz de pesos no negativos |
| Tamaño 3, 5 o 7 | Determina cuántos vecinos participan en el cálculo |
| Salto horizontal | Avance entre columnas de salida (1–8) |
| Salto vertical | Avance entre filas de salida (1–8) |
| **Ceros** | Añade un borde de ceros, como en la clase |
| **Reflejo** | Replica los valores cercanos mediante reflejo para evitar bordes oscuros |
| **Ver cálculo** | Explica con números de la foto la suma de los productos de la máscara |
| **Restaurar original** | Vuelve a mostrar la fotografía intacta en las cuatro vistas |
| **Procesar** | Aplica los ajustes visibles |
| **Guardar PNG** | Exporta solo los píxeles reales de la imagen reducida |

**Nota:** al seleccionar `Personalizada`, la caja de pesos se vuelve editable. Se escribe una fila por línea, con los números separados por espacios. Por ejemplo:

```text
1 1 1
1 4 1
1 1 1
```

El programa divide todos los pesos entre su suma (10). No acepta pesos negativos, para mantener la actividad enfocada en suavizado.

## 3. La idea matemática en cinco pasos

1. **Leer la fotografía.** Una fotografía RGB se guarda en una matriz de `alto × ancho × 3`.
2. **Crear la máscara.** Por ejemplo, la gaussiana 3×3: `[[1,2,1],[2,4,2],[1,2,1]] / 16`.
3. **Agregar padding.** Una máscara 3×3 usa un borde de un píxel; una máscara 5×5 usa dos.
4. **Aplicar la convolución.** Para cada posición se multiplica el píxel de la vecindad por el peso correspondiente, se suman los productos y se redondea el resultado. El algoritmo recorre las posiciones definidas por los saltos.
5. **Guardar la matriz reducida.** Un salto 2×2 toma un resultado por cada dos posiciones verticales y horizontales, después de combinar la vecindad con la máscara.

La convolución (para un canal) se define como:

\[
G(i,j)=\sum_{u=-r}^{r}\sum_{v=-r}^{r} F(i-u,j-v)\,M(u,v),
\quad r=\frac{K-1}{2}.
\]

Los pesos se invierten para respetar la definición matemática. Esto no altera máscaras simétricas como gaussiana o promedio.

La dimensión reducida con padding y centros desde cero es:

\[
H_{salida}=\left\lceil H/s_v\right\rceil,\qquad
W_{salida}=\left\lceil W/s_h\right\rceil.
\]

La reducción de cantidad de píxeles es:

\[
R=100\left(1-\frac{H_{salida}W_{salida}}{HW}\right)\%.
\]

### Ejemplo comprobado

- Imagen original: **626 × 417 = 261.042 píxeles**.
- Máscara: **gaussiana 3×3**.
- Salto horizontal: **2**.
- Salto vertical: **2**.
- Imagen reducida: **313 × 209 = 65.417 píxeles**.
- Reducción de cantidad de píxeles: **74,94 %**.

La proporción de píxeles no equivale necesariamente a la reducción del tamaño del archivo en bytes.

## 4. Por qué hay dos tipos de reducción

**Directa (sin filtro):** selecciona un píxel de cada bloque de posiciones; conserva dimensiones de salida, pero puede provocar aliasing en texturas o estructuras finas.

**Con suavizado (resultado principal):** combina los vecinos mediante una máscara antes de efectuar los saltos. Generalmente reduce el aliasing, a costa de cierto desenfoque. No se garantiza que la imagen conserve toda su información: reducir resolución siempre descarta muestras.

Para imágenes de muchos detalles y saltos grandes, conviene una máscara mayor y comparar los resultados. También se recomienda evaluar la opción de borde **Reflejo** si el padding de ceros crea un borde oscuro visible.

## 5. Cómo se logró una interfaz fluida sin complicar la matemática

El algoritmo principal está en `procesamiento.py` y solo contiene **dos bucles `for` sobre la máscara**. Para cada peso, NumPy multiplica una región de píxeles de una sola vez. Es matemáticamente equivalente a multiplicar y sumar las vecindades una por una, pero evita bucles lentos sobre cientos de miles de píxeles.

**No se emplean** `cv2.resize`, `PIL.Image.resize` para el algoritmo, `cv2.GaussianBlur`, `cv2.filter2D`, `scipy.ndimage.convolve`, `scipy.ndimage.zoom` ni ecualización automática.

**Única excepción para la visualización:** `Image.resize()` se utiliza dentro de `actualizar_panel()` para adaptar la vista previa al espacio de Tkinter. No cambia ni recalcula la imagen original, suavizada o reducida, y **jamás se utiliza para generar el PNG exportado**. Esta distinción se puede señalar durante la defensa.

## 6. Organización

```text
Taller_Suavizado_Reduccion_Python/
  main.py                 # Ventanas, botones, imágenes y controles
  procesamiento.py        # Máscara, padding, convolución y saltos
  pruebas.py              # Cálculos y pruebas matemáticas
  prueba_interfaz.py      # Pruebas reales de Tkinter
  requirements.txt        # NumPy y Pillow
  README.md               # Esta explicación
  DEFENSA_DOS_INTEGRANTES.md
  imagenes/
    paisaje.png
  evidencias/
    captura_interfaz.png
    comparacion_cuatro_resultados.png
    original.png
    suavizada.png
    reduccion_directa.png
    reduccion_suavizada.png
```

La ventana requiere un entorno gráfico de escritorio. No hay bases de datos, API, servidor, frameworks web ni clases creadas por el proyecto.

## 7. Limitaciones y transparencia

- Las reducciones por saltos grandes pueden presentar aliasing incluso con una máscara 7×7.
- El padding de ceros puede oscurecer los límites; el reflejo es una alternativa opcional.
- Los resultados se calculan con `float32` y se redondean a `uint8`.
- El tiempo mostrado corresponde al cálculo en la computadora que ejecuta el programa, no a una promesa fija de tiempo real.
- La calidad perceptual depende de la fotografía. El programa ofrece comparación directa, no una medición universal de belleza o calidad.
