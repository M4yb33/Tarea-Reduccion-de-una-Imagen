"""Pruebas integrales de interfaz Tkinter y cálculo paso a paso.

Verifica la versión de tres vistas, la resolución del recorte de la matriz 7x7 en el panel lateral,
la sincronización robusta de la ventana de cálculo paso a paso ante cambios de máscara, saltos e imágenes
(Casos A, B, C, D, E, F y G), la normalización matemática de pesos personalizados, la precisión de clics
en bordes y la exportación PNG.

Ejecutar: python prueba_interfaz.py
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import main
from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, calcular_geometria_panel,
                            convertir_clic_a_pixel, es_posicion_muestreada)


def sincronizar():
    main.raiz.update_idletasks()
    main.raiz.update()
    sleep(0.02)


class EventoSimulado:
    def __init__(self, x, y):
        self.x = x
        self.y = y


print('======================================================================')
print('EJECUTANDO PRUEBAS DE INTERFAZ Y CÁLCULO PASO A PASO')
print('======================================================================\n')

# -----------------------------------------------------------------------------
# 1. Verificación de las TRES vistas y dimensiones reales
# -----------------------------------------------------------------------------
raiz = main.construir_interfaz()
raiz.geometry('1440x920')
ruta_paisaje = Path(__file__).parent / 'imagenes' / 'paisaje.png'
main.cargar_imagen(str(ruta_paisaje))
sincronizar()

assert len(main.paneles) == 3, f"Deben existir exactamente 3 vistas, encontradas: {len(main.paneles)}"
assert set(main.paneles.keys()) == {'original', 'directa', 'reducida'}, "Claves incorrectas en paneles"
assert not hasattr(main, 'imagen_suavizada') or main.imagen_suavizada is None, "imagen_suavizada no debe mantenerse en el estado"
assert main.imagen_original.shape == (417, 626, 3), "Dimensiones originales incorrectas"
assert main.imagen_directa.shape == (209, 313, 3), "Dimensiones de reducción directa incorrectas"
assert main.imagen_reducida.shape == (209, 313, 3), "Dimensiones de reducción suavizada incorrectas"
print('GUI 1: Interfaz configurada exactamente con 3 vistas (Original, Reducción directa, Reducción con suavizado)')

# -----------------------------------------------------------------------------
# 2. Deslizadores de saltos y tiempo real sin reprocesamiento innecesario
# -----------------------------------------------------------------------------
salida_inicial = main.imagen_reducida.copy()
main.salto_horizontal.set(3)
main.actualizar_reduccion()
sincronizar()
assert main.imagen_reducida.shape == (209, 209, 3), f"Salto horizontal 3 falló: {main.imagen_reducida.shape}"

main.salto_vertical.set(3)
main.actualizar_reduccion()
sincronizar()
assert main.imagen_reducida.shape == (139, 209, 3), f"Salto vertical 3 falló: {main.imagen_reducida.shape}"

main.salto_horizontal.set(2)
main.salto_vertical.set(2)
main.actualizar_reduccion()
sincronizar()
assert np.array_equal(main.imagen_reducida, salida_inicial), "Reproducibilidad de reducción 2x2 falló"
print('GUI 2: Deslizadores horizontal y vertical actualizan ambas reducciones en tiempo real (209x313)')

# -----------------------------------------------------------------------------
# 3. Cambio de máscara y preservación de original
# -----------------------------------------------------------------------------
orig_backup = main.imagen_original.copy()
main.tipo_mascara.set('Promedio')
main.actualizar_mascara()
sincronizar()
assert np.array_equal(main.imagen_original, orig_backup), "La imagen original fue alterada al cambiar de máscara"
assert not np.array_equal(main.imagen_reducida, salida_inicial), "La reducción suavizada debió recalcularse"
print('GUI 3: Cambio a máscara de Promedio recalcula reducción con suavizado preservando el original intacto')

# -----------------------------------------------------------------------------
# 4. CORRECCIÓN 1 — Visibilidad completa y geometría real de matrices 3x3, 5x5 y 7x7 en barra lateral
# -----------------------------------------------------------------------------
# Comprobación de que 3x3, 5x5 y 7x7 caben perfectamente en c_box sin recorte
c_box = main.marco_celdas.master
ancho_disp = c_box.winfo_width() if c_box.winfo_width() > 1 else 308

for modo in ('Gaussiana', 'Personalizada'):
    main.tipo_mascara.set(modo)
    for n in (3, 5, 7):
        main.tamano_mascara.set(str(n))
        main.actualizar_mascara()
        sincronizar()
        req_w = main.marco_celdas.winfo_reqwidth()
        assert len(main.celdas_matriz) == n and len(main.celdas_matriz[0]) == n, f"Deben existir {n}x{n} celdas en {modo}"
        assert req_w <= ancho_disp, f"Matriz {n}x{n} ({modo}) sobresale: reqwidth {req_w} > disp {ancho_disp}"
        # Verificar visibilidad y posición de la última casilla (columna n-1)
        ultima_celda = main.celdas_matriz[0][n - 1]
        sincronizar()
        pos_fin = ultima_celda.winfo_x() + ultima_celda.winfo_width()
        assert pos_fin <= req_w + 4, f"Última celda {n}x{n} recortada en {modo}: pos {pos_fin} > req {req_w}"
        # Verificar que el contenido se lee correctamente
        txt_celda = ultima_celda.get().strip()
        assert len(txt_celda) > 0, "La celda no debe estar vacía"

print('GUI 4: CORRECCIÓN 1 VERIFICADA — Matrices 3x3, 5x5 y 7x7 (49 celdas) se muestran completas sin recorte')

# -----------------------------------------------------------------------------
# 5. Máscara personalizada identidad coincide con reducción directa
# -----------------------------------------------------------------------------
main.tipo_mascara.set('Personalizada')
main.tamano_mascara.set('3')
main.actualizar_mascara()
for f in range(3):
    for c in range(3):
        main.celdas_matriz[f][c].delete(0, 'end')
        main.celdas_matriz[f][c].insert(0, '1' if f == 1 and c == 1 else '0')
main.procesar()
sincronizar()
assert np.array_equal(main.imagen_reducida, main.imagen_directa), "Máscara impulso central debe coincidir con reducción directa"
print('GUI 5: Máscara personalizada de identidad equivale matemáticamente a la reducción directa')

# -----------------------------------------------------------------------------
# 6. Exportación PNG de resolución real (313x209)
# -----------------------------------------------------------------------------
with TemporaryDirectory() as tmp:
    dest = str(Path(tmp) / 'prueba_export.png')
    orig_dialog = main.filedialog.asksaveasfilename
    main.filedialog.asksaveasfilename = lambda **kw: dest
    try:
        main.guardar_png()
    finally:
        main.filedialog.asksaveasfilename = orig_dialog
    leida = np.asarray(Image.open(dest).convert('RGB'))
    assert leida.shape == (209, 313, 3), f"Dimensiones de PNG exportado incorrectas: {leida.shape}"
    assert np.array_equal(leida, main.imagen_reducida), "Datos numéricos del PNG difieren de la matriz interna"
print('GUI 6: Guardado PNG conserva dimensiones exactas (313x209) y datos de reducción con suavizado')

# -----------------------------------------------------------------------------
# 7. Botón Restaurar
# -----------------------------------------------------------------------------
main.salto_horizontal.set(4)
main.salto_vertical.set(4)
main.tipo_mascara.set('Promedio')
main.restaurar()
sincronizar()
assert main.tipo_mascara.get() == 'Gaussiana', "Restaurar debe dejar Gaussiana"
assert main.tamano_mascara.get() == '3', "Restaurar debe dejar 3x3"
assert main.salto_horizontal.get() == 2 and main.salto_vertical.get() == 2, "Restaurar debe dejar salto 2x2"
assert main.imagen_reducida.shape == (209, 313, 3), "Restaurar debe recalcular a 313x209"
print('GUI 7: Botón Restaurar restablece parámetros iniciales (Gaussiana 3x3, salto 2x2, 313x209)')

# -----------------------------------------------------------------------------
# 8. Rechazo estricto de máscara con ceros en la interfaz
# -----------------------------------------------------------------------------
main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
for f in range(3):
    for c in range(3):
        main.celdas_matriz[f][c].delete(0, 'end')
        main.celdas_matriz[f][c].insert(0, '0')
try:
    main.mascara_desde_interfaz()
    assert False, "Debió rechazar máscara de ceros"
except ValueError as e:
    assert 'al menos un peso positivo' in str(e)
print('GUI 8: Validación de interfaz rechaza máscara personalizada de ceros sin sustitución silenciosa')

# -----------------------------------------------------------------------------
# 9. Clics en centro, esquinas, bordes y márgenes con geometría real
# -----------------------------------------------------------------------------
lz_orig, _ = main.paneles['original']
sincronizar()
geom = calcular_geometria_panel(lz_orig.winfo_width(), lz_orig.winfo_height(), 626, 417, margen_interior=8)

# Centro
main.manejar_clic_original(EventoSimulado(geom['cx'], geom['cy']))
assert main.pixel_seleccionado is not None
f_c, c_c = main.pixel_seleccionado
assert abs(f_c - 208) <= 2 and abs(c_c - 313) <= 2, f"Centro incorrecto: ({f_c}, {c_c})"

# Borde izquierdo exacto
main.manejar_clic_original(EventoSimulado(geom['ox'], geom['cy']))
assert main.pixel_seleccionado[1] == 0, f"Borde izquierdo debe ser col 0, obtenido: {main.pixel_seleccionado[1]}"

# Borde derecho exacto (con tolerancia de 1 píxel por resolución inferior de pantalla)
main.manejar_clic_original(EventoSimulado(geom['ox'] + geom['vw'] - 1, geom['cy']))
assert abs(main.pixel_seleccionado[1] - 625) <= 1, f"Borde derecho debe aproximar a col 625, obtenido: {main.pixel_seleccionado[1]}"

# Borde superior exacto
main.manejar_clic_original(EventoSimulado(geom['cx'], geom['oy']))
assert main.pixel_seleccionado[0] == 0, f"Borde superior debe ser fila 0, obtenido: {main.pixel_seleccionado[0]}"

# Borde inferior exacto (con tolerancia de 1 píxel por resolución inferior de pantalla)
main.manejar_clic_original(EventoSimulado(geom['cx'], geom['oy'] + geom['vh'] - 1))
assert abs(main.pixel_seleccionado[0] - 416) <= 1, f"Borde inferior debe aproximar a fila 416, obtenido: {main.pixel_seleccionado[0]}"

# Margen fuera de la imagen (debe ignorarse sin cambiar selección)
prev_sel = main.pixel_seleccionado
main.manejar_clic_original(EventoSimulado(geom['ox'] - 10, geom['cy']))
assert main.pixel_seleccionado == prev_sel, "Clic fuera no debe alterar selección previa"
print(f'GUI 9: Selección por clic en centro ({f_c}, {c_c}), 4 bordes y márgenes verificada exactamente')

# -----------------------------------------------------------------------------
# 10. Cálculo paso a paso: normalización de pesos con suma 1.25 en imagen uniforme 100
# -----------------------------------------------------------------------------
main.restaurar()
sincronizar()

img_100 = np.full((50, 50, 3), 100, dtype=np.uint8)
main.imagen_original = img_100
main.tipo_mascara.set('Personalizada')
main.tamano_mascara.set('3')
pesos_125 = [[0.125, 0.125, 0.125],
             [0.125, 0.250, 0.125],
             [0.125, 0.125, 0.125]]
assert np.isclose(sum(sum(f) for f in pesos_125), 1.25)
main.establecer_matriz_codigo(pesos_125)
main.procesar()
sincronizar()

main.mostrar_calculo()
sincronizar()
v_calc = main.ventana_calculo_actual
assert v_calc is not None

main.callback_actualizar_calculo(25, 25)
sincronizar()

body = [w for w in v_calc.winfo_children() if isinstance(w, main.tk.Frame) and w != v_calc.winfo_children()[0]][0]
lbl_sum = [w for w in body.winfo_children() if isinstance(w, main.tk.Label) and 'Suma ponderada' in w.cget('text')][0]

texto_suma = lbl_sum.cget('text')
assert 'Píxel final = 100' in texto_suma, f"El píxel final debió ser 100 tras normalizar: {texto_suma}"
assert '125' not in texto_suma, f"Error: se usaron pesos sin normalizar (se obtuvo 125): {texto_suma}"

pixel_procesado = main.imagen_reducida[12, 12, 0]
assert pixel_procesado == 100, f"La imagen reducida procesada debió tener valor 100, obtenido: {pixel_procesado}"
print('GUI 10: Normalización en cálculo paso a paso probada (intensidad 100 con pesos suma 1.25 -> resultado 100, no 125)')

# -----------------------------------------------------------------------------
# 11. CORRECCIÓN 2 — CASOS A, B, C, D, E, F y G de sincronización de cálculo paso a paso
# -----------------------------------------------------------------------------
main.cargar_imagen(str(ruta_paisaje))
sincronizar()

# --- Caso A: 3x3 a 7x7 con cálculo abierto y selección de nuevo píxel ---
main.tipo_mascara.set('Gaussiana')
main.tamano_mascara.set('3')
main.actualizar_mascara()
main.mostrar_calculo()
sincronizar()
v_act = main.ventana_calculo_actual
assert v_act is not None and v_act.winfo_exists()
assert '3x3' in v_act.title()

# Cambiar a 7x7 en ventana principal
main.tamano_mascara.set('7')
main.actualizar_mascara()
sincronizar()
v_7 = main.ventana_calculo_actual
assert v_7 is not None and v_7.winfo_exists(), "La ventana de cálculo debió actualizarse a 7x7"
assert '7x7' in v_7.title(), f"Título de ventana incorrecto tras cambio a 7x7: {v_7.title()}"

# Seleccionar otro píxel y verificar que no hay excepciones
main.manejar_clic_original(EventoSimulado(geom['cx'] + 10, geom['cy'] + 10))
sincronizar()
# Contar casillas en la ventana de cálculo: deben ser 7x7 = 49 en cada bloque
b_7 = [w for w in v_7.winfo_children() if isinstance(w, main.tk.Frame) and w != v_7.winfo_children()[0]][0]
c_mat_7 = [w for w in b_7.winfo_children() if isinstance(w, main.tk.Frame) and len(w.winfo_children()) >= 1][0]
cv_7 = [w for w in c_mat_7.winfo_children() if isinstance(w, main.tk.Canvas)][0]
mr_7 = [w for w in cv_7.winfo_children() if isinstance(w, main.tk.Frame)][0]
cuadros_7 = [w for w in mr_7.winfo_children() if isinstance(w, main.tk.Frame)]
for blk in range(3):
    sub = cuadros_7[blk].winfo_children()[1]
    assert len(sub.winfo_children()) == 49, f"Bloque {blk} debe tener 49 celdas en 7x7"
print('GUI 11: Caso A probado — Cambio 3x3 a 7x7 sincroniza 49 celdas y permite seleccionar píxeles sin excepción')

# --- Caso B: 7x7 a 3x3 en sentido inverso ---
main.tamano_mascara.set('3')
main.actualizar_mascara()
sincronizar()
v_3 = main.ventana_calculo_actual
assert v_3 is not None and v_3.winfo_exists()
assert '3x3' in v_3.title()
b_3 = [w for w in v_3.winfo_children() if isinstance(w, main.tk.Frame) and w != v_3.winfo_children()[0]][0]
c_mat_3 = [w for w in b_3.winfo_children() if isinstance(w, main.tk.Frame) and len(w.winfo_children()) >= 1][0]
cv_3 = [w for w in c_mat_3.winfo_children() if isinstance(w, main.tk.Canvas)][0]
mr_3 = [w for w in cv_3.winfo_children() if isinstance(w, main.tk.Frame)][0]
cuadros_3 = [w for w in mr_3.winfo_children() if isinstance(w, main.tk.Frame)]
for blk in range(3):
    sub = cuadros_3[blk].winfo_children()[1]
    assert len(sub.winfo_children()) == 9, f"Bloque {blk} debe tener 9 celdas en 3x3"
print('GUI 12: Caso B probado — Cambio 7x7 a 3x3 sincroniza 9 celdas en sentido inverso')

# --- Caso C: Cambio de tipo (Gaussiana a Personalizada) manteniendo tamaño ---
main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
sincronizar()
v_p = main.ventana_calculo_actual
assert v_p is not None and v_p.winfo_exists()
b_p = [w for w in v_p.winfo_children() if isinstance(w, main.tk.Frame) and w != v_p.winfo_children()[0]][0]
c_mat_p = [w for w in b_p.winfo_children() if isinstance(w, main.tk.Frame) and len(w.winfo_children()) >= 1][0]
cv_p = [w for w in c_mat_p.winfo_children() if isinstance(w, main.tk.Canvas)][0]
mr_p = [w for w in cv_p.winfo_children() if isinstance(w, main.tk.Frame)][0]
cuadros_p = [w for w in mr_p.winfo_children() if isinstance(w, main.tk.Frame)]
sub_pes_p = cuadros_p[1].winfo_children()[1]
for ew in sub_pes_p.winfo_children():
    assert ew.cget('state') == 'normal', "Las celdas de pesos deben ser editables en Personalizada"
print('GUI 13: Caso C probado — Cambio de tipo a Personalizada actualiza permisos de edición a normal')

# --- Caso D: Cambio de saltos con ventana abierta actualiza indicador de muestreo sin reconstruir ---
id_antes = id(main.ventana_calculo_actual)
lbl_salto_d = [w for w in b_p.winfo_children() if isinstance(w, main.tk.Label) and 'muestrea' in w.cget('text')][0]
main.callback_actualizar_calculo(0, 0)
sincronizar()
assert 'SÍ se muestrea' in lbl_salto_d.cget('text')

main.salto_horizontal.set(4)
main.salto_vertical.set(4)
main.actualizar_reduccion()
sincronizar()
assert id(main.ventana_calculo_actual) == id_antes, "La ventana de cálculo no debe destruirse por cambio de saltos"
assert 'salto 4x4' in lbl_salto_d.cget('text'), "El indicador de salto debe reflejar el nuevo salto 4x4"
print('GUI 14: Caso D probado — Cambio de saltos actualiza indicador en tiempo real sin destruir la ventana')

# --- Caso E: Cambio de imagen con ventana abierta ---
with TemporaryDirectory() as tmp_img_dir:
    ruta_img2 = Path(tmp_img_dir) / 'test_chg.png'
    Image.fromarray(np.full((80, 80, 3), 120, dtype=np.uint8)).save(ruta_img2)
    main.cargar_imagen(str(ruta_img2))
    sincronizar()
    v_e = main.ventana_calculo_actual
    assert v_e is not None and v_e.winfo_exists()
    assert main.imagen_original.shape == (80, 80, 3)
    assert main.pixel_seleccionado == (40, 40)
print('GUI 15: Caso E probado — Cargar nueva imagen sincroniza dimensiones y píxel central en cálculo paso a paso')

# --- Caso F: Cierre manual con X y reapertura limpia ---
main.cargar_imagen(str(ruta_paisaje))
sincronizar()
main.cerrar_calculo_seguro()
assert main.ventana_calculo_actual is None
assert main.callback_actualizar_calculo is None
assert main.callback_actualizar_salto is None
main.tamano_mascara.set('5')
main.actualizar_mascara()
sincronizar()
assert main.ventana_calculo_actual is None, "Si la ventana estaba cerrada, cambiar máscara no debe abrirla automáticamente"
main.mostrar_calculo()
sincronizar()
assert main.ventana_calculo_actual is not None and '5x5' in main.ventana_calculo_actual.title()
print('GUI 16: Caso F probado — Cierre manual limpia referencias; reapertura posterior funciona sin callbacks obsoletos')

# --- Caso G: Aplicación de máscara personalizada desde la propia ventana ---
v_g = main.ventana_calculo_actual
b_g = [w for w in v_g.winfo_children() if isinstance(w, main.tk.Frame) and w != v_g.winfo_children()[0]][0]
c_mat_g = [w for w in b_g.winfo_children() if isinstance(w, main.tk.Frame) and len(w.winfo_children()) >= 1][0]
cv_g = [w for w in c_mat_g.winfo_children() if isinstance(w, main.tk.Canvas)][0]
mr_g = [w for w in cv_g.winfo_children() if isinstance(w, main.tk.Frame)][0]
cuadros_g = [w for w in mr_g.winfo_children() if isinstance(w, main.tk.Frame)]
sub_pes_g = cuadros_g[1].winfo_children()[1]

# Cambiar a Personalizada 3x3 para prueba asimétrica
main.tipo_mascara.set('Personalizada')
main.tamano_mascara.set('3')
main.actualizar_mascara()
sincronizar()

v_g3 = main.ventana_calculo_actual
b_g3 = [w for w in v_g3.winfo_children() if isinstance(w, main.tk.Frame) and w != v_g3.winfo_children()[0]][0]
c_mat_g3 = [w for w in b_g3.winfo_children() if isinstance(w, main.tk.Frame) and len(w.winfo_children()) >= 1][0]
cv_g3 = [w for w in c_mat_g3.winfo_children() if isinstance(w, main.tk.Canvas)][0]
mr_g3 = [w for w in cv_g3.winfo_children() if isinstance(w, main.tk.Frame)][0]
cuadros_g3 = [w for w in mr_g3.winfo_children() if isinstance(w, main.tk.Frame)]
sub_pes_g3 = cuadros_g3[1].winfo_children()[1]
entries_g3 = sub_pes_g3.winfo_children()

# Configurar máscara asimétrica: [1.0, 0, 0; 0, 0, 0; 0, 0, 0]
for idx, ew in enumerate(entries_g3):
    ew.delete(0, 'end')
    ew.insert(0, '1.0' if idx == 0 else '0.0')

top_bar_g = v_g3.winfo_children()[0]
btn_recalc_g = [w for w in top_bar_g.winfo_children() if isinstance(w, main.tk.Button) and w.cget('text') == 'Recalcular'][0]
btn_recalc_g.invoke()
sincronizar()

# Simular confirmación en "Aplicar máscara a la imagen"
main.messagebox.askyesno = lambda *a, **kw: True
main.messagebox.showinfo = lambda *a, **kw: None
btn_aplicar_g = [w for w in top_bar_g.winfo_children() if isinstance(w, main.tk.Button) and 'Aplicar' in w.cget('text')][0]
btn_aplicar_g.invoke()
sincronizar()

# Comprobar que la ventana no se destruyó inesperadamente
assert main.ventana_calculo_actual is not None and main.ventana_calculo_actual.winfo_exists(), "La ventana no debe destruirse al aplicar la máscara desde su botón"

# Comprobar coincidencia matemática entre cálculo local y global
f_eval, c_eval = 20, 20
main.callback_actualizar_calculo(f_eval, c_eval)
sincronizar()
# Máscara asimétrica con 1.0 en esquina sup-izq convoluciona con el píxel inf-der (f+1, c+1)
val_esperado_g = int(main.imagen_original[f_eval + 1, c_eval + 1, 0])
salida_global_g = int(main.imagen_reducida[f_eval // int(main.salto_vertical.get()), c_eval // int(main.salto_horizontal.get()), 0])
lbl_sum_g = [w for w in b_g3.winfo_children() if isinstance(w, main.tk.Label) and 'Suma ponderada' in w.cget('text')][0]
assert f'Píxel final = {val_esperado_g}' in lbl_sum_g.cget('text'), "Cálculo local paso a paso asimétrico falló"
assert salida_global_g == val_esperado_g, f"Fallo global asimétrico: esperado {val_esperado_g}, obtenido {salida_global_g}"
print(f'GUI 17: Caso G probado — Aplicación de máscara desde cálculo no destruye ventana y coincide matemáticamente ({val_esperado_g})')

# -----------------------------------------------------------------------------
# 18. Rechazo de entradas inválidas en ventana paso a paso
# -----------------------------------------------------------------------------
entries_g3[0].delete(0, 'end')
btn_recalc_g.invoke()
sincronizar()
assert 'Edición pendiente' in lbl_sum_g.cget('text'), f"Esperado estado pendiente: {lbl_sum_g.cget('text')}"

for ew in entries_g3:
    ew.delete(0, 'end')
    ew.insert(0, '0')
btn_recalc_g.invoke()
sincronizar()
assert 'suma de los pesos debe ser mayor a 0' in lbl_sum_g.cget('text'), f"Esperado error suma cero: {lbl_sum_g.cget('text')}"

entries_g3[0].configure(validate='none')
entries_g3[0].delete(0, 'end')
entries_g3[0].insert(0, 'invalido')
btn_recalc_g.invoke()
sincronizar()
assert 'Peso no numérico' in lbl_sum_g.cget('text'), f"Esperado error peso no numérico: {lbl_sum_g.cget('text')}"
print('GUI 18: Ventana de cálculo paso a paso presenta estado pendiente/aviso ante entradas inválidas sin cerrarse ni sustituir por ceros')

main.cerrar_calculo_seguro()

# -----------------------------------------------------------------------------
# 19. Verificación de CSV y métricas sin referencias obsoletas
# -----------------------------------------------------------------------------
with TemporaryDirectory() as tmpdir:
    ruta_csv = Path(tmpdir) / 'test_metricas.csv'
    datos_test = main.calcular_datos(main.imagen_original, main.imagen_reducida, 15.0)
    main.exportar_metricas_csv(str(ruta_csv), 'paisaje.png', datos_test, 'Gaussiana', 3, 2, 2)
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        contenido_csv = f.read()
    assert 'Media Reducida' in contenido_csv, "CSV debe contener Media Reducida"
    assert 'Media Suavizada' not in contenido_csv, "CSV no debe contener referencias a Media Suavizada completa"
    assert 'Variacion Reducida' in contenido_csv, "CSV debe contener Variacion Reducida"
print('GUI 19: Exportación CSV verificada sin referencias obsoletas a imágenes inexistentes')

# -----------------------------------------------------------------------------
# 20. Verificación de presets personalizados ('1s', '1-2-1' Gaussiana binomial y 'Centro 5')
# -----------------------------------------------------------------------------
main.tipo_mascara.set('Personalizada')

# 3x3: Preset '1-2-1'
main.tamano_mascara.set('3')
main.actualizar_mascara()
main.aplicar_preset('gauss')
sincronizar()
assert main.celdas_matriz[1][1].get() == '4', f"Centro 3x3 '1-2-1' debió ser 4, obtenido {main.celdas_matriz[1][1].get()}"
assert main.celdas_matriz[0][0].get() == '1', "Esquina (0,0) 3x3 debió ser 1"
assert main.celdas_matriz[0][1].get() == '2', "Celda (0,1) 3x3 debió ser 2"

# 5x5: Preset '1-2-1'
main.tamano_mascara.set('5')
main.actualizar_mascara()
main.aplicar_preset('gauss')
sincronizar()
assert main.celdas_matriz[2][2].get() == '36', f"Centro 5x5 '1-2-1' debió ser 36, obtenido {main.celdas_matriz[2][2].get()}"
assert main.celdas_matriz[0][0].get() == '1', "Esquina (0,0) 5x5 debió ser 1"
assert main.celdas_matriz[0][1].get() == '4', "Celda (0,1) 5x5 debió ser 4"

# 7x7: Preset '1-2-1' - Comprobación crítica: centro debe ser 400, NUNCA 5
main.tamano_mascara.set('7')
main.actualizar_mascara()
main.aplicar_preset('gauss')
sincronizar()
assert main.celdas_matriz[3][3].get() == '400', f"Centro 7x7 '1-2-1' debió ser 400, obtenido {main.celdas_matriz[3][3].get()} (error del '5' en el medio)"
assert main.celdas_matriz[0][0].get() == '1', "Esquina (0,0) 7x7 debió ser 1"
assert main.celdas_matriz[0][1].get() == '6', "Celda (0,1) 7x7 debió ser 6"
assert main.celdas_matriz[0][2].get() == '15', "Celda (0,2) 7x7 debió ser 15"
assert main.celdas_matriz[0][3].get() == '20', "Celda (0,3) 7x7 debió ser 20"
assert main.celdas_matriz[3][2].get() == '300', "Celda (3,2) 7x7 debió ser 300 (15x20)"

# 7x7: Preset 'Centro 5' - Aquí sí debe ser 5 en el centro y 1 en el resto
main.aplicar_preset('centro')
sincronizar()
assert main.celdas_matriz[3][3].get() == '5', "Centro 7x7 'Centro 5' debió ser 5"
assert main.celdas_matriz[0][0].get() == '1', "Esquina (0,0) 7x7 debió ser 1"
assert main.celdas_matriz[0][1].get() == '1', "Celda (0,1) 7x7 debió ser 1"

# 7x7: Preset '1s' - Todos 1s
main.aplicar_preset('unos')
sincronizar()
for f in range(7):
    for c in range(7):
        assert main.celdas_matriz[f][c].get() == '1', f"Celda ({f},{c}) debió ser 1 en preset 'unos'"

print('GUI 20: Presets personalizados verificados — Preset 1-2-1 en 7x7 genera pesos binomiales con centro 400 (corregido el defecto del 5 central)')

raiz.destroy()
print('\n======================================================================')
print('TODAS LAS 20 PRUEBAS DE INTERFAZ Y CÁLCULO FINALIZARON CON ÉXITO.')
print('======================================================================')
