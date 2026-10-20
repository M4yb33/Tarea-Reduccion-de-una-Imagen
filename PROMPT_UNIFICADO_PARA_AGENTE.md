# Prompt unificado: taller de suavizado y reducción de imágenes

Desarrollar un programa de escritorio completo en **Python** para el segundo parcial de Inteligencia Artificial de la Universidad Técnica de Ambato. La tarea consiste en **reducir imágenes digitales mediante convolución manual y saltos configurables**. Debe permitir elegir una máscara de suavizado impar, mostrar sus pesos, introducir saltos horizontal y vertical, visualizar el antes y después, medir el cambio de dimensiones y guardar el resultado.

## Requisitos prioritarios

1. **El código debe ser fácil de comprender, estudiar y defender por dos estudiantes.** Evitar arquitecturas elaboradas, clases innecesarias, APIs, bases de datos, modelos de IA, concurrencia, recursividad y expresiones crípticas. Dar preferencia a funciones breves, variables en español y comentarios matemáticos.
2. **El algoritmo debe implementar por sí mismo** el padding, los productos máscara × vecindad, la suma de estos productos, el avance por saltos y la reconstrucción RGB. No usar `cv2.resize`, `PIL.Image.resize`, `skimage.transform.resize`, `scipy.ndimage.zoom`, `cv2.pyrDown`, `cv2.blur`, `cv2.GaussianBlur`, `cv2.filter2D`, `scipy.ndimage.convolve` ni funciones que hagan de forma automática el filtrado o la reducción de salida.
3. **NumPy está permitido** para representar matrices, recorrer por cortes, realizar multiplicaciones y sumas vectorizadas cuando no se oculta el algoritmo. `Image.resize` solo se permite para **visualizar** una imagen en la interfaz, nunca para generar ni guardar el resultado reducido.
4. Ofrecer máscaras de **Promedio**, **Gaussiana** y **Personalizada**, tamaños **3×3, 5×5 y 7×7**, con pesos normalizados. Gaussiana 3×3 de referencia: `[[1,2,1],[2,4,2],[1,2,1]]/16`. Explicar el producto exterior de `[1,2,1]` consigo mismo.
5. Incluir **padding de ceros** como opción predeterminada de clase y **padding por reflejo** como comparación. Los píxeles se procesan por centros de ventana sobre la imagen original.
6. Configurar por separado **salto horizontal 1–8** y **salto vertical 1–8**, predeterminados en 2×2. Para relleno que conserva los centros, las dimensiones serán `ceil(alto/salto_vertical)` y `ceil(ancho/salto_horizontal)`.
7. Mostrar cuatro resultados: original, suavizada a resolución original, reducción directa sin filtro y reducción después del suavizado. Los cuatro paneles deben incluir dimensiones reales. El PNG exportado debe mantener el tamaño reducido real.
8. La interfaz **Tkinter** debe presentar controles intuitivos, matriz visible, actualización al mover los saltos, comparación visual amplia, tiempos medidos y cambio porcentual de píxeles.
9. Añadir un diálogo **«Ver cálculo paso a paso»** que muestre, para una posición real, la vecindad de cada canal RGB, los pesos, la multiplicación y la suma que produce el valor resultante.
10. Implementar pruebas reproducibles: máscara normalizada, ejemplo central conocido (110), padding, dimensiones y saltos asimétricos, RGB uint8, convolución de máscara asimétrica, guardado exacto y controles GUI.
11. Documentar honestamente que **toda reducción de resolución descarta información**, que el suavizado busca reducir aliasing y que un filtro demasiado intenso genera desenfoque. No prometer calidad óptima ni un rendimiento que no se haya medido.

## Estructura mínima

`main.py` para interfaz, `procesamiento.py` para el algoritmo, `pruebas.py` para comprobar fórmulas y `prueba_interfaz.py` para comprobar que los controles funcionan; además `requirements.txt`, imagen de ejemplo, README y guía para defensa de dos estudiantes.

## Orden de prioridad

**Corrección matemática > sencillez de explicación > buena calidad perceptual y comparación > fluidez de la interfaz > documentación > optimización avanzada.**

El producto se debe entregar **implementado, ejecutado, probado y con una captura auténtica de la interfaz**, no solo como pseudocódigo o maqueta visual.
