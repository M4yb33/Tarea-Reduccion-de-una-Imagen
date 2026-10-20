"""Generador automatizado de evidencias para el Taller de IA.

Genera imágenes de comparación, capturas de pantalla de la interfaz y de la
ventana de cálculo paso a paso, y guarda los reportes de pruebas en UTF-8.
"""

import os
import sys
import subprocess
from pathlib import Path
import ctypes
from ctypes import wintypes
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
EVIDENCIAS_DIR = BASE_DIR / 'evidencias'
EVIDENCIAS_DIR.mkdir(exist_ok=True)

import main
from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, calcular_datos)


def capturar_widget_gdi(widget, ruta_salida):
    """Captura el contenido de una ventana Tkinter mediante Windows GDI PrintWindow."""
    widget.update()
    hwnd = widget.winfo_id()
    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        w = widget.winfo_width()
        h = widget.winfo_height()

    hdc_win = ctypes.windll.user32.GetWindowDC(hwnd)
    hdc_mem = ctypes.windll.gdi32.CreateCompatibleDC(hdc_win)
    hbm = ctypes.windll.gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    ctypes.windll.gdi32.SelectObject(hdc_mem, hbm)

    ctypes.windll.user32.PrintWindow(hwnd, hdc_mem, 2)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', wintypes.DWORD),
            ('biWidth', wintypes.LONG),
            ('biHeight', wintypes.LONG),
            ('biPlanes', wintypes.WORD),
            ('biBitCount', wintypes.WORD),
            ('biCompression', wintypes.DWORD),
            ('biSizeImage', wintypes.DWORD),
            ('biXPelsPerMeter', wintypes.LONG),
            ('biYPelsPerMeter', wintypes.LONG),
            ('biClrUsed', wintypes.DWORD),
            ('biClrImportant', wintypes.DWORD)
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0

    buf = (ctypes.c_char * (w * h * 4))()
    ctypes.windll.gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)
    im = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1).convert('RGB')

    ctypes.windll.gdi32.DeleteObject(hbm)
    ctypes.windll.gdi32.DeleteDC(hdc_mem)
    ctypes.windll.user32.ReleaseDC(hwnd, hdc_win)

    im.save(ruta_salida)
    print(f'[OK] Captura guardada en {ruta_salida.name} ({w}x{h} px)')


def generar_imagenes_procesamiento():
    """Genera las 4 imágenes individuales y el mosaico comparativo."""
    ruta_img = BASE_DIR / 'imagenes' / 'paisaje.png'
    img_orig = np.asarray(Image.open(ruta_img).convert('RGB'))
    masc_gauss = crear_mascara('Gaussiana', 3)

    img_suav = aplicar_convolucion(img_orig, masc_gauss, 1, 1)
    img_dir = reducir_sin_suavizado(img_orig, 2, 2)
    img_red = aplicar_convolucion(img_orig, masc_gauss, 2, 2)

    Image.fromarray(img_orig).save(EVIDENCIAS_DIR / 'original.png')
    Image.fromarray(img_suav).save(EVIDENCIAS_DIR / 'suavizada.png')
    Image.fromarray(img_dir).save(EVIDENCIAS_DIR / 'reduccion_directa.png')
    Image.fromarray(img_red).save(EVIDENCIAS_DIR / 'reduccion_suavizada.png')

    # Mosaico comparativo
    ancho_panel, alto_panel = 626, 417
    mosaico = Image.new('RGB', (ancho_panel * 2 + 30, alto_panel * 2 + 90), color=(240, 243, 248))
    dibujo = ImageDraw.Draw(mosaico)

    # Pegar 4 vistas
    im_o = Image.fromarray(img_orig)
    im_s = Image.fromarray(img_suav)
    im_d = Image.fromarray(img_dir).resize((ancho_panel, alto_panel), Image.Resampling.NEAREST)
    im_r = Image.fromarray(img_red).resize((ancho_panel, alto_panel), Image.Resampling.LANCZOS)

    mosaico.paste(im_o, (10, 30))
    mosaico.paste(im_s, (ancho_panel + 20, 30))
    mosaico.paste(im_d, (10, alto_panel + 70))
    mosaico.paste(im_r, (ancho_panel + 20, alto_panel + 70))

    dibujo.text((15, 10), '1. IMAGEN ORIGINAL (626x417 px - 261,042 pixeles)', fill=(24, 36, 58))
    dibujo.text((ancho_panel + 25, 10), '2. IMAGEN SUAVIZADA (626x417 px - Gaussiana 3x3)', fill=(24, 36, 58))
    dibujo.text((15, alto_panel + 50), '3. REDUCCIÓN DIRECTA (313x209 px - Sin filtro: Aliasing)', fill=(180, 83, 9))
    dibujo.text((ancho_panel + 25, alto_panel + 50), '4. REDUCCIÓN CON SUAVIZADO (313x209 px - Anti-Aliasing)', fill=(5, 150, 105))

    mosaico.save(EVIDENCIAS_DIR / 'comparacion_cuatro_resultados.png')
    print('[OK] Mosaico y 4 imágenes generados en evidencias/')


def capturar_interfaz_y_calculo():
    """Ejecuta la interfaz, carga la imagen y toma capturas de pantalla."""
    raiz = main.construir_interfaz()
    main.cargar_imagen(str(BASE_DIR / 'imagenes' / 'paisaje.png'))
    raiz.update()
    capturar_widget_gdi(raiz, EVIDENCIAS_DIR / 'captura_interfaz.png')

    main.mostrar_calculo()
    raiz.update()
    v_calc = main.ventana_calculo_actual
    if v_calc:
        v_calc.update()
        capturar_widget_gdi(v_calc, EVIDENCIAS_DIR / 'captura_calculo_paso_a_paso.png')

    raiz.destroy()


def ejecutar_y_guardar_pruebas():
    """Ejecuta pruebas.py y prueba_interfaz.py guardando su salida en UTF-8."""
    res_pruebas = subprocess.run(
        [sys.executable, str(BASE_DIR / 'pruebas.py')],
        capture_output=True, text=True, encoding='utf-8'
    )
    with open(EVIDENCIAS_DIR / 'resultados_pruebas.txt', 'w', encoding='utf-8') as f:
        f.write(res_pruebas.stdout)
    print(f'[OK] resultados_pruebas.txt generado (exit code: {res_pruebas.returncode})')

    res_gui = subprocess.run(
        [sys.executable, str(BASE_DIR / 'prueba_interfaz.py')],
        capture_output=True, text=True, encoding='utf-8'
    )
    with open(EVIDENCIAS_DIR / 'resultados_interfaz.txt', 'w', encoding='utf-8') as f:
        f.write(res_gui.stdout)
    print(f'[OK] resultados_interfaz.txt generado (exit code: {res_gui.returncode})')


if __name__ == '__main__':
    print('=== GENERANDO EVIDENCIAS COMPLETAS DEL PROYECTO ===')
    generar_imagenes_procesamiento()
    capturar_interfaz_y_calculo()
    ejecutar_y_guardar_pruebas()
    print('=== TODAS LAS EVIDENCIAS GENERADAS CON ÉXITO ===')
