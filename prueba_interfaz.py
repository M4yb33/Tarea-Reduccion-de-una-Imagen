"""Prueba real de Tkinter. Requiere escritorio o xvfb-run en Linux."""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
import numpy as np
from PIL import Image
import main


def esperar():
    for _ in range(75):
        main.raiz.update()
        if main.actualizacion_pendiente is None:
            return
        sleep(0.02)
    raise AssertionError('Los controles no completaron la actualización.')


raiz = main.construir_interfaz()
ruta = Path(__file__).parent / 'imagenes' / 'paisaje.png'
main.cargar_imagen(str(ruta))
esperar()
assert main.imagen_original.shape == (417, 626, 3)
assert main.imagen_reducida.shape == (209, 313, 3)
assert main.imagen_directa.shape == main.imagen_reducida.shape
assert len(main.paneles) == 4
print('GUI 1: carga y muestra cuatro fotografías con dimensiones verificadas')

salida_inicial = main.imagen_reducida.copy()
main.salto_horizontal.set(3)
main.solicitar_procesamiento()
esperar()
assert main.imagen_reducida.shape == (209, 209, 3)
print('GUI 2: deslizador horizontal actualiza resolución sin pulsar Procesar')

main.salto_vertical.set(3)
main.solicitar_procesamiento()
esperar()
assert main.imagen_reducida.shape == (139, 209, 3)
print('GUI 3: deslizador vertical actualiza resolución sin pulsar Procesar')

main.salto_horizontal.set(2)
main.salto_vertical.set(2)
main.solicitar_procesamiento()
esperar()
assert np.array_equal(main.imagen_reducida, salida_inicial)
print('GUI 4: volver a los mismos valores reproduce la imagen píxel por píxel')

main.tipo_mascara.set('Promedio')
main.actualizar_mascara()
esperar()
assert not np.array_equal(main.imagen_reducida, salida_inicial)
print('GUI 5: cambiar máscara recalcula el resultado')

main.tipo_mascara.set('Personalizada')
main.actualizar_mascara()
main.editor_mascara.delete('1.0', 'end')
main.editor_mascara.insert('1.0', '0 0 0\n0 1 0\n0 0 0')
main.procesar()
esperar()
assert np.array_equal(main.imagen_reducida, main.imagen_directa)
print('GUI 6: máscara personalizada de identidad equivale a reducción directa')

with TemporaryDirectory() as tmp:
    dest = str(Path(tmp) / 'foto_reducida.png')
    original_dialog = main.filedialog.asksaveasfilename
    main.filedialog.asksaveasfilename = lambda **kw: dest
    try:
        main.guardar_png()
    finally:
        main.filedialog.asksaveasfilename = original_dialog
    assert np.array_equal(np.asarray(Image.open(dest).convert('RGB')),
                          main.imagen_reducida)
print('GUI 7: guardado PNG conserva dimensiones y valores reales')

main.mostrar_calculo()
raiz.update()
ventanas = [v for v in raiz.winfo_children() if isinstance(v, main.tk.Toplevel)]
assert ventanas
for ventana in ventanas:
    ventana.destroy()
print('GUI 8: calculadora de convolución con valores reales funciona')

main.restaurar()
esperar()
assert np.array_equal(main.imagen_reducida, main.imagen_original)
assert main.resumen_reduccion.get().startswith('0.0 %')
print('GUI 9: Restaurar original recupera exactamente la foto de entrada')

main.procesar()
esperar()
assert main.imagen_reducida.shape == (209, 313, 3)
print('GUI 10: procesar después de restaurar usa los parámetros visibles')

raiz.destroy()
print('TODAS LAS PRUEBAS DE INTERFAZ FINALIZADAS.')
