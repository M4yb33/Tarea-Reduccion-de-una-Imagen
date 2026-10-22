# Taller de Inteligencia Artificial: Suavizado y Reducción de Imágenes

**Universidad Técnica de Ambato — Segundo Parcial**  
**Proyecto didáctico, funcional y fácil de explicar para la defensa de dos estudiantes.**

El software permite cargar fotografías RGB, añadir ruido artificial, aplicar filtros impares (3×3, 5×5 o 7×7), editar pesos personalizados y reducir la resolución espacial controlando los saltos (*strides*) horizontal y vertical. Se presentan **tres vistas comparativas equilibradas**:
1. **IMAGEN ORIGINAL:** entrada RGB a resolución completa, con el ruido seleccionado y selección interactiva de píxel mediante clic.
2. **REDUCCIÓN DIRECTA (SIN SUAVIZADO):** píxeles muestreados directamente mediante saltos (`imagen[::sv, ::sh]`), como referencia de comparación y aliasing.
3. **REDUCCIÓN CON SUAVIZADO:** resultado principal obtenido aplicando la máscara de convolución normalizada a las vecindades en las posiciones de salida definidas por los saltos.

Ninguna función prediseñada realiza el filtrado o la reducción (`cv2.resize`, `cv2.GaussianBlur`, `PIL.Image.resize` no se usan para el cómputo matricial). Todo el procesamiento matemático se implementa con bucles explícitos y NumPy en `procesamiento.py`.

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
Ejecuta las **18 pruebas matemáticas y de equivalencia obligatorias** del proyecto, finalizando con código de salida 0.

Para pruebas automatizadas integrales de la interfaz Tkinter y del cálculo interactivo:
```bash
python prueba_interfaz.py
```
Ejecuta las **20 pruebas de interfaz**, validando visibilidad de 7×7 sin recorte, Casos A al G de sincronización de cálculo paso a paso, normalización de pesos, precisión de clics, presets ('1-2-1' en 7×7) y las 3 vistas.

Para regenerar evidencias y capturas actualizadas:
```bash
python generar_evidencias.py
```

---

## 2. Organización Visual y Controles de la Interfaz

La interfaz principal organiza las **tres vistas** de forma equilibrada: la fotografía original grande a la izquierda ocupando toda la altura, y las dos imágenes reducidas apiladas verticalmente a la derecha para comparar con comodidad el efecto del suavizado frente a la reducción directa.

| Control | Descripción |
|---|---|
| **Cargar imagen** | Abre una fotografía RGB sin distorsionar sus dimensiones numéricas reales. |
| **Procesar** | Normaliza la máscara seleccionada, calcula las reducciones directa y suavizada, y registra resultados en `resultados_procesamiento.csv`. Evita calcular innecesariamente imágenes suavizadas a resolución completa. |
| **Restaurar** | Restablece la fotografía original y los parámetros por defecto (Gaussiana 3×3, salto 2×2). |
| **Guardar PNG** | Exporta la matriz reducida con suavizado real a un archivo PNG sin pérdidas ni escalados artificiales de visualización. |
| **Clic en Imagen Original** | Mapeo unificado mediante `calcular_geometria_panel()` que utiliza los anchos y altos enteros efectivamente representados (`vw, vh`) y los desplazamientos reales (`ox, oy`), garantizando selección coherente en centro, bordes y esquinas. |
| **Máscara Gaussiana** | Coeficientes discretos normalizados generados mediante producto exterior $[1, 2, 1] \times [1, 2, 1] / 16.0$ para 3×3. Campos de solo lectura. |
| **Máscara Promedio** | Pesos uniformes normalizados ($1/K^2$). Campos de solo lectura. |
| **Filtro Moda** | Calcula manualmente el valor más frecuente de cada canal RGB en la vecindad. Es especialmente útil para eliminar ruido de sal y pimienta. |
| **Máscara Personalizada** | **Única máscara editable**: cuadrícula independiente con validación estricta (rechaza entradas vacías, negativas o con suma cero sin sustitución silenciosa) y presets rápidos (`1s`, `1-2-1`, `Centro 5`). |
| **Ruido artificial (imnoise)** | Permite elegir **Gaussiano** o **Sal y pimienta**. En gaussiano, el control representa la varianza sobre la escala normalizada; en sal y pimienta, representa la densidad de píxeles alterados. `0` no añade ruido y un valor cercano a `1` degrada casi toda la imagen. |
| **Saltos de Reducción** | Deslizadores horizontal y vertical (1 a 8). Actualizan ambas reducciones en tiempo real. |
| **Resultados del Procesamiento** | Tarjeta lateral compacta con resolución original, reducida, conteo total de píxeles, porcentaje de reducción y tiempo de cálculo en ms. |
| **Ver cálculo paso a paso** | Abre la ventana didáctica para inspeccionar $I \times W_{\text{conv}} = P \rightarrow \text{Suma} \rightarrow \text{Píxel de salida}$ calculada con normalización matemática estricta y precisión flotante completa (`float64`). |

---

## 3. Fundamento Matemático

1. **Lectura RGB:** Una imagen se representa como un tensor NumPy de dimensiones `(alto, ancho, 3)` con tipo `uint8`.
2. **Ruido artificial:** Antes de filtrar, se crea una copia de la imagen cargada. El ruido gaussiano suma una muestra normal de media cero y varianza configurada; el ruido sal y pimienta reemplaza píxeles seleccionados aleatoriamente por negro o blanco. La imagen original cargada se conserva para evitar acumular ruido al cambiar el control.
3. **Padding de Ceros:** Se añade un borde de ceros alrededor de la imagen de grosor $p = (K - 1) / 2$ para centrar el kernel en los bordes.
   - Máscara 3×3: $p = 1$.
   - Máscara 5×5: $p = 2$.
   - Máscara 7×7: $p = 3$.
   *Nota académica:* En las esquinas y bordes extremos, parte de las muestras son ceros artificiales, lo cual produce un oscurecimiento leve y matemáticamente normal.
4. **Convolución Matemática con Saltos:** La máscara se normaliza para que $\sum W_{ij} = 1.0$ y se invierte en ambos ejes ($W[::-1, ::-1]$). Los centros avanzan de $s_v$ en $s_v$ filas y $s_h$ en $s_h$ columnas:
   $$H_{\text{salida}} = \left\lfloor\frac{H - 1}{s_v}\right\rfloor + 1, \qquad W_{\text{salida}} = \left\lfloor\frac{W - 1}{s_h}\right\rfloor + 1$$
5. **Porcentaje de Reducción:**
   $$R = \left(1 - \frac{N_{\text{final}}}{N_{\text{original}}}\right) \times 100\%$$
   Para la imagen de prueba `626 × 417` con salto `2×2`:
   $$\text{Original: } 626 \times 417 = 261.042 \text{ px} \longrightarrow \text{Final: } 313 \times 209 = 65.417 \text{ px} \quad (-74,94\%)$$

---

## 4. Ventana de Cálculo Paso a Paso

Titulada **Cálculo paso a paso — Convolución**, permite:
- Seleccionar coordenadas `Fila`, `Columna` y canal (`Rojo`, `Verde`, `Azul`).
- Comprobar la **relación con el salto**: indica si la posición seleccionada forma parte del submuestreo de la reducción ($fila \bmod s_v == 0$ y $columna \bmod s_h == 0$).
- **Matriz de Píxeles (I):** Carga los valores reales de la vecindad y permite editarlos para demostración manual sin alterar la imagen global.
- **Matriz de Máscara (W):** Muestra los pesos normalizados (solo lectura para Gaussiana/Promedio; editable para Personalizada).
- **Normalización estricta de pesos:** Si el usuario edita pesos personalizados cuya suma difiere de 1.0 (por ejemplo suma 1.25 en una región uniforme de 100), la ventana los normaliza automáticamente mediante `crear_mascara()`, produciendo el resultado correcto (100, no 125).
- **Convolución con inversión:** Aplica la orientación efectiva $W[::-1, ::-1]$ para que máscaras asimétricas coincidan de forma idéntica en el ejercicio local y en el filtrado global.
- **Adaptabilidad 3×3, 5×5 y 7×7:** Dimensionamiento dinámico de fuentes y casillas con contenedor desplazable horizontalmente, evitando cualquier desbordamiento de pantalla.
- Botón **Aplicar máscara a la imagen**: Aplica los pesos editados confirmados a la imagen completa sin dobles inversiones erróneas.

---

## 5. Estructura del Código

```text
Tarea-Reduccion-de-una-Imagen/
├── main.py                  # Interfaz Tkinter simplificada a 3 vistas y eventos
├── procesamiento.py         # Convolución manual, saltos, normalización y métricas
├── pruebas.py               # Las 18 pruebas obligatorias + equivalencias (exit code 0)
├── prueba_interfaz.py       # Las 20 pruebas de Tkinter y cálculo paso a paso (exit code 0)
├── generar_evidencias.py    # Generador automatizado de capturas y reportes
├── DEFENSA_DOS_INTEGRANTES.md # Guía de estudio y guion de exposición para 2 estudiantes
├── requirements.txt         # Dependencias mínimas (numpy, pillow)
├── imagenes/                # Fotografías de prueba (paisaje.png, paisaje.jpg)
└── evidencias/              # Capturas de 3 vistas, cálculo 3x3, 7x7 y resultados TXT
```
