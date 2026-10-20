# Taller de Inteligencia Artificial: Suavizado y Reducción de Imágenes

**Universidad Técnica de Ambato — Segundo Parcial**
**Proyecto didáctico, funcional y fácil de explicar para dos estudiantes.**

El software permite cargar fotografías RGB, aplicar máscaras impares de convolución (3×3, 5×5 o 7×7), editar pesos personalizados y reducir la resolución controlando los saltos (*strides*) horizontal y vertical. Se presentan cuatro vistas comparativas sincronizadas:
1. **Imagen original** (con selección de píxel mediante clic interactivo).
2. **Imagen suavizada** (convolución completa a resolución 1×1).
3. **Reducción directa** (submuestreo sin filtro, como referencia de aliasing).
4. **Reducción con suavizado** (convolución manual + saltos).

Ninguna función prediseñada realiza el filtrado o la reducción (`cv2.resize`, `cv2.GaussianBlur`, `PIL.Image.resize` no se usan para el cálculo matemático). Todo el procesamiento matemático se implementa con bucles explícitos y NumPy en `procesamiento.py`.

---

## 1. Instalación y Ejecución

Se requiere **Python 3.10 o posterior** con `numpy` y `pillow`.

```bash
python -m pip install -r requirements.txt
python main.py
```

El programa carga automáticamente `imagenes/paisaje.png` si está disponible. Se puede cargar cualquier otra fotografía PNG, JPG o BMP con el botón **Cargar imagen**.

### Ejecución de Pruebas Automatizadas:
```bash
python pruebas.py
```
Ejecuta las **18 pruebas matemáticas y de interfaz obligatorias** del proyecto, finalizando con código de salida 0.

Para pruebas automatizadas de la interfaz Tkinter:
```bash
python prueba_interfaz.py
```

---

## 2. Controles de la Interfaz

| Control | Descripción |
|---|---|
| **Cargar imagen** | Abre una fotografía RGB sin alterar sus dimensiones originales. |
| **Procesar** | Único botón de procesamiento: normaliza pesos, convoluciona, actualiza las 4 vistas y registra resultados en `resultados_procesamiento.csv`. |
| **Restaurar** | Restablece la fotografía original y los parámetros por defecto (Gaussiana 3×3, salto 2×2). |
| **Guardar PNG** | Exporta la matriz reducida exacta a un archivo PNG sin pérdidas ni escalados artificiales. |
| **Clic en Imagen Original** | Mapeo unificado mediante `calcular_geometria_panel()` que sincroniza el margen interior de 8 px del lienzo con la matriz real de la foto, garantizando selección exacta sin desfasajes de filas ni columnas. |
| **Máscara Gaussiana** | Pesos exactos generados mediante el producto exterior $[1, 2, 1] \times [1, 2, 1] / 16.0$ para 3×3. Campos de solo lectura. |
| **Máscara Promedio** | Pesos uniformes normalizados ($1/K^2$). Campos de solo lectura. |
| **Máscara Personalizada** | **Única máscara editable**: cuadrícula independiente con validación estricta (rechaza entradas vacías, negativas o con suma cero sin sustitución silenciosa) y presets rápidos (`1s`, `1-2-1`, `Centro 5`). |
| **Saltos de Reducción** | Deslizadores horizontal y vertical (1 a 8). Actualizan la reducción en tiempo real. |
| **Resultados del Procesamiento** | Tarjeta lateral compacta con resolución original, reducida, conteo total de píxeles, porcentaje de reducción y tiempo de cálculo en ms. |
| **Ver cálculo paso a paso** | Abre la ventana didáctica para inspeccionar $I \times W = P \rightarrow \text{Suma} \rightarrow \text{Píxel de salida}$ calculada con precisión flotante completa (`float64`), coincidiendo al 100% con la convolución general. |

---

## 3. Fundamento Matemático

1. **Lectura RGB:** Una imagen se representa como un tensor NumPy de dimensiones `(alto, ancho, 3)` con tipo `uint8`.
2. **Padding de Ceros:** Se añade un borde de ceros alrededor de la imagen de grosor $p = (K - 1) / 2$ para centrar el kernel en los bordes.
   - Máscara 3×3: $p = 1$.
   - Máscara 5×5: $p = 2$.
   - Máscara 7×7: $p = 3$.
   *Nota:* El padding de ceros puede atenuar u oscurecer ligeramente las intensidades en los bordes extremos, lo cual es matemáticamente normal y esperado.
3. **Convolución:** Se invierten los coeficientes de la máscara $W[::-1, ::-1]$ (en máscaras simétricas es idéntica). Se recorre cada vecindad multiplicando el píxel por el peso y sumando los productos.
4. **Muestreo por Saltos (Strides):** Los centros de convolución avanzan de $s_v$ en $s_v$ filas y de $s_h$ en $s_h$ columnas.
   - Dimensiones resultantes:
     $$H_{\text{salida}} = \left\lfloor\frac{H - 1}{s_v}\right\rfloor + 1, \qquad W_{\text{salida}} = \left\lfloor\frac{W - 1}{s_h}\right\rfloor + 1$$
5. **Porcentaje de Reducción:**
   $$R = \left(1 - \frac{N_{\text{final}}}{N_{\text{original}}}\right) \times 100\%$$
   Para la imagen de prueba `626 × 417` con salto `2×2`:
   $$\text{Original: } 261.042 \text{ px} \longrightarrow \text{Final: } 313 \times 209 = 65.417 \text{ px} \quad (-74,94\%)$$

---

## 4. Ventana de Cálculo Paso a Paso

Titulada **Cálculo paso a paso — Convolución**, permite:
- Seleccionar coordenadas `Fila`, `Columna` y canal (`Rojo`, `Verde`, `Azul`).
- Comprobar la **relación con el salto**: indica si la posición seleccionada forma parte del submuestreo de la reducción ($fila \bmod s_v == 0$ y $columna \bmod s_h == 0$).
- **Matriz de Píxeles (I):** Carga los valores reales de la vecindad y permite editarlos para ejercicios matemáticos locales sin alterar la imagen global.
- **Matriz de Máscara (W):** Muestra los pesos normalizados (solo lectura para Gaussiana/Promedio; editable para Personalizada).
- **Matriz de Productos (I × W):** Muestra cada producto individual y la suma ponderada $S = \sum I_{ij} W_{ij}$, junto con el píxel final redondeado y acotado en `[0, 255]`.
- Botón **Aplicar máscara a la imagen**: Si se modificaron pesos personalizados, permite confirmarlos y aplicarlos a la fotografía completa.

---

## 5. Estructura del Código

```text
Tarea-Reduccion-de-una-Imagen/
├── main.py                  # Interfaz gráfica Tkinter y eventos de usuario
├── procesamiento.py         # Funciones matemáticas de convolución, saltos y métricas
├── pruebas.py               # Las 18 pruebas obligatorias (código de salida 0)
├── prueba_interfaz.py       # Pruebas automatizadas de la GUI Tkinter
├── DEFENSA_DOS_INTEGRANTES.md # Guía de estudio y guion para los dos estudiantes
├── requirements.txt         # Dependencias mínimas (numpy, pillow)
├── imagenes/                # Fotografías de prueba (paisaje.png, paisaje.jpg)
└── evidencias/              # Capturas y resultados verificables generados
```
