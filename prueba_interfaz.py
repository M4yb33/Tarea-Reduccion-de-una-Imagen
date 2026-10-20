"""Prueba integral de la interfaz Tkinter para el Taller de IA."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import main


def sincronizar():
    main.raiz.update_idletasks()
    main.raiz.update()
    sleep(0.02)


raiz = main.construir_interfaz()
raiz.geometry('1440x920')
ruta = Path(__file__).parent / 'imagenes' / 'paisaje.png'
main.cargar_imagen(str(ruta))
sincronizar()

assert main.imagen_original.shape == (417, 626, 3)
assert main.imagen_reducida.shape == (209, 313, 3)
assert main.imagen_directa.shape == main.imagen_reducida.shape
assert len(main.paneles) == 4
print('GUI 1: Carga y muestra cuatro fotografías con dimensiones verificadas (626x417 -> 313x209)')

salida_inicial = main.imagen_reducida.copy()
main.salto_horizontal.set(3)
main.actualizar_reduccion()
sincronizar()
assert main.imagen_reducida.shape == (209, 209, 3)
print('GUI 2: Deslizador horizontal actualiza resolución en tiempo real a 209x209')

main.salto_vertical.set(3)
main.actualizar_reduccion()
sincronizar()
assert main.imagen_reducida.shape == (139, 209, 3)
print('GUI 3: Deslizador vertical actualiza resolución en tiempo real a 209x139')

main.salto_horizontal.set(2)
main.salto_vertical.set(2)
main.actualizar_reduccion()
sincronizar()
assert np.array_equal(main.imagen_reducida, salida_inicial)
print('GUI 4: Volver a los mismos saltos (2x2) reproduce la imagen exactamente píxel por píxel')

main.tipo_mascara.set('Promedio')
main.actualizar_mascara()
sincronizar()
assert not np.array_equal(main.imagen_reducida, salida_inicial)
print('GUI 5: Cambiar a máscara de Promedio recalcula el resultado de reducción')

# Máscara personalizada de identidad: centro 1, resto 0
main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
for f in range(3):
    for c in range(3):
        main.celdas_matriz[f][c].delete(0, 'end')
        main.celdas_matriz[f][c].insert(0, '1' if f == 1 and c == 1 else '0')
main.procesar()
sincronizar()
# Máscara de impulso unitario central equivale a muestreo directo
assert np.array_equal(main.imagen_reducida, main.imagen_directa)
print('GUI 6: Máscara personalizada identidad equivale matemáticamente a reducción directa')

# Exportación a PNG
with TemporaryDirectory() as tmp:
    dest = str(Path(tmp) / 'foto_reducida.png')
    original_dialog = main.filedialog.asksaveasfilename
    main.filedialog.asksaveasfilename = lambda **kw: dest
    try:
        main.guardar_png()
    finally:
        main.filedialog.asksaveasfilename = original_dialog
    assert np.array_equal(np.asarray(Image.open(dest).convert('RGB')), main.imagen_reducida)
print('GUI 7: Guardado PNG conserva dimensiones y valores reales sin pérdida')

# Ventana de cálculo paso a paso
main.mostrar_calculo()
sincronizar()
ventanas = [v for v in raiz.winfo_children() if isinstance(v, main.tk.Toplevel)]
assert ventanas
for ventana in ventanas:
    ventana.destroy()
print('GUI 8: Ventana interactiva de cálculo paso a paso se inicializa y abre correctamente')

# Botón Restaurar
main.salto_horizontal.set(5)
main.salto_vertical.set(5)
main.tipo_mascara.set('Promedio')
main.restaurar()
sincronizar()
assert main.tipo_mascara.get() == 'Gaussiana'
assert main.salto_horizontal.get() == 2
assert main.salto_vertical.get() == 2
assert main.imagen_reducida.shape == (209, 313, 3)
print('GUI 9: Botón Restaurar restablece configuración inicial (Gaussiana 3x3, salto 2x2)')

# GUI 10: Validación estricta sin sustitución silenciosa de máscara inválida
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
    assert main.celdas_matriz[0][0].get() == '0'
print('GUI 10: Máscara personalizada de ceros es rechazada sin sustitución silenciosa')

# GUI 11: Selección por clic en el lienzo sin desfasaje
lienzo_orig, _ = main.paneles['original']
geom_orig = main.calcular_geometria_panel(lienzo_orig.winfo_width(), lienzo_orig.winfo_height(), 626, 417, margen_interior=8)
class EventoSimulado:
    def __init__(self, x, y):
        self.x = x
        self.y = y
ev_centro = EventoSimulado(geom_orig['cx'], geom_orig['cy'])
main.manejar_clic_original(ev_centro)
assert main.pixel_seleccionado is not None
f_sel, c_sel = main.pixel_seleccionado
assert abs(f_sel - 208) <= 2 and abs(c_sel - 313) <= 2
print(f'GUI 11: Selección por clic en lienzo mapea correctamente a píxel ({f_sel}, {c_sel})')

# GUI 12: Coincidencia matemática exacta en cálculo paso a paso
main.restaurar()
sincronizar()
main.mostrar_calculo()
sincronizar()
v_calc = main.ventana_calculo_actual
assert v_calc is not None
main.callback_actualizar_calculo(106, 576)
sincronizar()
valor_esperado = main.imagen_suavizada[106, 576, 1]
reg, pes, _, res_pix = main.explicar_pixel(main.imagen_original, main.mascara_actual, 106, 576)
assert res_pix[1] == valor_esperado
v_calc.destroy()
print(f'GUI 12: Ventana de cálculo paso a paso coincide exactamente con imagen suavizada ({valor_esperado})')

raiz.destroy()
print('TODAS LAS PRUEBAS DE INTERFAZ FINALIZADAS CON ÉXITO.')
