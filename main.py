"""Taller de IA — Suavizado y Reducción de Imágenes.

Universidad Técnica de Ambato.
Convolución manual con máscaras y reducción controlada por saltos (strides).
"""

import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, calcular_datos, explicar_pixel,
                            exportar_metricas_csv, calcular_geometria_panel,
                            convertir_clic_a_pixel, es_posicion_muestreada)

# Paleta de colores
FONDO, BLANCO, TEXTO, SECUNDARIO = '#eef2f7', '#ffffff', '#18243a', '#64748b'
AZUL, VERDE, FONDO_IMG, ROJO = '#2458dc', '#059669', '#141c29', '#dc2626'

# Matriz personalizada por defecto editable en código
MATRIZ_PERSONALIZADA_DEFECTO = [
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0]
]
matriz_personalizada_guardada = [list(f) for f in MATRIZ_PERSONALIZADA_DEFECTO]

# Variables globales de estado
raiz = None
imagen_original = imagen_suavizada = imagen_reducida = imagen_directa = None
mascara_actual = datos_actuales = None
ruta_actual = ''
pixel_seleccionado = None

# Variables de control de interfaz
nombre_imagen = tipo_mascara = tamano_mascara = None
salto_horizontal = salto_vertical = paso_x_etiqueta = paso_y_etiqueta = None
estado = explicacion_mascara = lbl_integracion_texto = lbl_pixel_info = None
marco_celdas = marco_herramientas_pers = None
paneles, fotos_tk, celdas_matriz = {}, {}, []

# Referencias de la tarjeta de resultados del procesamiento
lbl_res_orig = lbl_res_red = lbl_pix_tot = lbl_pct_red = None
lbl_cfg_masc = lbl_cfg_saltos = lbl_t_proc = None

# Referencia a ventana de cálculo paso a paso
ventana_calculo_actual = None
callback_actualizar_calculo = None


def validar_decimal(texto):
    """Permite exclusivamente números: dígitos y como máximo un punto o coma decimal."""
    if texto == '' or texto in ('.', ','):
        return True
    if any(c not in '0123456789.,' for c in texto):
        return False
    return texto.count('.') + texto.count(',') <= 1


def validar_pixel(texto):
    """Permite exclusivamente dígitos enteros de 0 a 255 para intensidades RGB."""
    return texto == '' or (texto.isdigit() and 0 <= int(texto) <= 255)


def validar_entero_coord(texto):
    """Permite exclusivamente dígitos enteros no negativos para coordenadas de fila/columna."""
    return texto == '' or texto.isdigit()


def establecer_matriz_codigo(nueva_matriz):
    """Permite definir o cambiar la matriz personalizada directamente desde código."""
    global matriz_personalizada_guardada
    matriz_personalizada_guardada = [list(f) for f in nueva_matriz]
    if tipo_mascara and tipo_mascara.get() == 'Personalizada':
        actualizar_mascara()


def cargar_imagen(ruta=None):
    """Carga imagen RGB sin alterar su resolución real y la procesa."""
    global imagen_original, ruta_actual, pixel_seleccionado
    if not ruta:
        ruta = filedialog.askopenfilename(
            title='Seleccionar imagen',
            filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp')]
        )
    if not ruta:
        return
    try:
        with Image.open(ruta) as f:
            imagen_original = np.array(f.convert('RGB'), dtype=np.uint8)
        ruta_actual = str(ruta)
        pixel_seleccionado = (imagen_original.shape[0] // 2, imagen_original.shape[1] // 2)
        if nombre_imagen:
            nombre_imagen.set(Path(ruta).name)
        if lbl_pixel_info:
            lbl_pixel_info.config(text=f'Píxel seleccionado: fila {pixel_seleccionado[0]}, columna {pixel_seleccionado[1]}')
        procesar()
    except Exception as e:
        messagebox.showerror('Error al cargar imagen', str(e))


def guardar_png():
    """Exporta la matriz reducida real en formato PNG sin distorsiones."""
    if imagen_reducida is None:
        messagebox.showwarning('Sin resultado', 'Primero cargue y procese una imagen.')
        return
    ruta = filedialog.asksaveasfilename(
        title='Guardar imagen reducida',
        defaultextension='.png',
        filetypes=[('PNG', '*.png')]
    )
    if ruta:
        try:
            Image.fromarray(imagen_reducida).save(ruta)
            if estado:
                estado.set(f'Guardada con éxito: {Path(ruta).name} ({imagen_reducida.shape[1]}x{imagen_reducida.shape[0]} px)')
        except Exception as e:
            messagebox.showerror('Error al guardar', str(e))


def restaurar():
    """Restaura la fotografía original y los parámetros iniciales de la práctica."""
    global pixel_seleccionado
    if imagen_original is None:
        return
    tipo_mascara.set('Gaussiana')
    tamano_mascara.set('3')
    salto_horizontal.set(2)
    salto_vertical.set(2)
    paso_x_etiqueta.set('2')
    paso_y_etiqueta.set('2')
    pixel_seleccionado = (imagen_original.shape[0] // 2, imagen_original.shape[1] // 2)
    if lbl_pixel_info:
        lbl_pixel_info.config(text=f'Píxel seleccionado: fila {pixel_seleccionado[0]}, columna {pixel_seleccionado[1]}')
    actualizar_mascara()
    if estado:
        estado.set('Configuración inicial restaurada (Gaussiana 3x3, salto 2x2).')


def mascara_desde_interfaz():
    """Obtiene máscara normalizada según las opciones activas."""
    global matriz_personalizada_guardada
    tipo, tam = tipo_mascara.get(), int(tamano_mascara.get())
    if tipo != 'Personalizada':
        return crear_mascara(tipo, tam)
    valores = []
    for f in range(tam):
        fila = []
        for c in range(tam):
            if f >= len(celdas_matriz) or c >= len(celdas_matriz[f]):
                raise ValueError('La cuadrícula no coincide con el tamaño seleccionado.')
            txt = celdas_matriz[f][c].get().strip()
            if not txt:
                raise ValueError(f'La celda (fila {f+1}, columna {c+1}) está vacía. Ingrese un valor numérico.')
            try:
                num = float(txt.replace(',', '.'))
            except ValueError:
                raise ValueError(f'El valor "{txt}" en (fila {f+1}, columna {c+1}) no es un número válido.')
            if not np.isfinite(num):
                raise ValueError(f'El valor en (fila {f+1}, columna {c+1}) debe ser finito.')
            if num < 0:
                raise ValueError(f'El peso ({num}) en (fila {f+1}, columna {c+1}) es negativo. Solo se permiten pesos no negativos.')
            fila.append(num)
        valores.append(fila)
    matriz = np.asarray(valores, dtype=np.float64)
    if not np.any(matriz > 0):
        raise ValueError('La máscara personalizada debe tener al menos un peso positivo (la suma no puede ser cero).')
    matriz_personalizada_guardada = [list(f) for f in valores]
    return crear_mascara(tipo, tam, matriz)


def aplicar_preset(preset):
    """Carga pesos predefinidos en la cuadrícula personalizada."""
    tam = int(tamano_mascara.get())
    p = [[1, 2, 1], [2, 4, 2], [1, 2, 1]] if preset == 'gauss' and tam == 3 else None
    for f in range(tam):
        for c in range(tam):
            val = '1' if preset == 'unos' else ('5' if (f == tam // 2 and c == tam // 2) else '1')
            if p:
                val = str(p[f][c])
            celdas_matriz[f][c].delete(0, 'end')
            celdas_matriz[f][c].insert(0, val)
    procesar()


def actualizar_mascara(*args):
    """Reconstruye casillas de la máscara: editables en Personalizada, solo lectura en Promedio y Gaussiana."""
    global celdas_matriz
    if not marco_celdas:
        return
    tipo, tam = tipo_mascara.get(), int(tamano_mascara.get())
    for w in marco_celdas.winfo_children():
        w.destroy()
    celdas_matriz = []
    w_col = 5 if tam == 3 else (4 if tam == 5 else 3)
    vcmd_dec = (raiz.register(validar_decimal), '%P') if raiz else None

    if tipo == 'Personalizada':
        for f in range(tam):
            fila_w = []
            for c in range(tam):
                kwargs = {'validate': 'key', 'validatecommand': vcmd_dec} if vcmd_dec else {}
                e = tk.Entry(marco_celdas, width=w_col, justify='center', font=('Consolas', 10, 'bold'),
                             relief='solid', bd=1, **kwargs)
                e.grid(row=f, column=c, padx=2, pady=2)
                if f < len(matriz_personalizada_guardada) and c < len(matriz_personalizada_guardada[f]):
                    val = str(matriz_personalizada_guardada[f][c])
                else:
                    val = '1'
                e.insert(0, val)
                e.bind('<Return>', lambda ev: procesar())
                fila_w.append(e)
            celdas_matriz.append(fila_w)
        marco_herramientas_pers.pack(fill='x', pady=2)
        explicacion_mascara.set('Personalizada: edite los pesos en los cuadros y presione "Procesar".')
    else:
        marco_herramientas_pers.pack_forget()
        m = crear_mascara(tipo, tam)
        for f in range(tam):
            fila_w = []
            for c in range(tam):
                e = tk.Entry(marco_celdas, width=w_col + 2, justify='center', font=('Consolas', 9),
                             relief='solid', bd=1, bg='#f1f5f9', fg=SECUNDARIO)
                e.grid(row=f, column=c, padx=2, pady=2)
                e.insert(0, f'{m[f, c]:.4f}')
                e.configure(state='readonly')
                fila_w.append(e)
            celdas_matriz.append(fila_w)
        explicacion_mascara.set(f'Pesos normalizados ({tipo}): suma = {m.sum():.4f}')
    procesar()


def actualizar_reduccion(*args):
    """Calcula y visualiza la reducción con saltos en tiempo real y actualiza el resumen."""
    global imagen_reducida, imagen_directa, datos_actuales
    if imagen_original is None:
        return
    sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
    paso_x_etiqueta.set(str(sx))
    paso_y_etiqueta.set(str(sy))

    masc = mascara_actual if mascara_actual is not None else mascara_desde_interfaz()
    t0 = time.perf_counter()
    imagen_reducida = aplicar_convolucion(imagen_original, masc, sy, sx)
    imagen_directa = reducir_sin_suavizado(imagen_original, sy, sx)
    t_ms = (time.perf_counter() - t0) * 1000
    datos_actuales = calcular_datos(imagen_original, imagen_suavizada, imagen_reducida, t_ms)

    actualizar_panel('directa')
    actualizar_panel('reducida')
    actualizar_resumen_resultados(datos_actuales, sx, sy)

    if estado:
        nw, nh = imagen_reducida.shape[1], imagen_reducida.shape[0]
        pct = datos_actuales['reduccion']
        estado.set(f'[OK] Salto {sx}x{sy} -> {nw}x{nh} px | Reducción: {pct:.2f}% píxeles ({t_ms:.1f} ms)')


def actualizar_resumen_resultados(datos, sx, sy):
    """Actualiza la sección Resultados del Procesamiento en la barra lateral."""
    if not lbl_res_orig or not datos:
        return
    w_orig, h_orig = datos['ancho_original'], datos['alto_original']
    w_fin, h_fin = datos['ancho_final'], datos['alto_final']
    n_orig, n_fin = datos['pixeles_original'], datos['pixeles_final']
    pct = datos['reduccion']
    t_ms = datos['tiempo_ms']
    tipo = tipo_mascara.get()
    tam = tamano_mascara.get()

    lbl_res_orig.config(text=f'Original: {w_orig} x {h_orig} px')
    lbl_res_red.config(text=f'Reducida: {w_fin} x {h_fin} px')
    lbl_pix_tot.config(text=f'Píxeles: {n_orig:,} -> {n_fin:,}')
    lbl_pct_red.config(text=f'Reducción: {pct:.2f} %')
    lbl_cfg_masc.config(text=f'Máscara: {tipo} ({tam}x{tam})')
    lbl_cfg_saltos.config(text=f'Saltos: H: {sx}  |  V: {sy}')
    lbl_t_proc.config(text=f'Tiempo cálculo: {t_ms:.1f} ms')


def procesar():
    """Punto único de procesamiento: suavizado completo, reducción con saltos y CSV."""
    global mascara_actual, imagen_suavizada
    if imagen_original is None:
        if estado:
            estado.set('Cargue una fotografía para comenzar.')
        return
    try:
        mascara_actual = mascara_desde_interfaz()
        imagen_suavizada = aplicar_convolucion(imagen_original, mascara_actual, 1, 1)
        actualizar_panel('original')
        actualizar_panel('suavizada')
        actualizar_reduccion()
    except ValueError as ve:
        if estado:
            estado.set(f'[!] Máscara no válida: {ve}')
        messagebox.showerror('Máscara no válida', str(ve))
        return
    except Exception as e:
        if estado:
            estado.set(f'Error en procesamiento: {e}')
        messagebox.showerror('Error en procesamiento', str(e))
        return

    if tipo_mascara.get() == 'Personalizada' and lbl_integracion_texto:
        filas = ' | '.join(' '.join(f'{x:.3f}' for x in f) for f in mascara_actual)
        lbl_integracion_texto.config(text=f'Integrada: [{filas}] (suma={mascara_actual.sum():.2f})')
    try:
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        exportar_metricas_csv('resultados_procesamiento.csv', Path(ruta_actual).name,
                              datos_actuales, tipo_mascara.get(), int(tamano_mascara.get()), sx, sy)
    except Exception:
        pass


def actualizar_panel(clave):
    """Muestra la imagen adaptada al panel manteniendo la relación de aspecto."""
    img = {'original': imagen_original, 'suavizada': imagen_suavizada,
           'directa': imagen_directa, 'reducida': imagen_reducida}.get(clave)
    if img is None or imagen_original is None or clave not in paneles:
        return
    lienzo, info = paneles[clave]
    img_h, img_w = img.shape[:2]
    geom = calcular_geometria_panel(lienzo.winfo_width(), lienzo.winfo_height(), img_w, img_h, margen_interior=8)
    if geom is None:
        return

    vw, vh = geom['vw'], geom['vh']
    cx, cy = geom['cx'], geom['cy']
    foto = Image.fromarray(img)
    vista = foto.resize((vw, vh), Image.Resampling.LANCZOS)
    foto_tk = ImageTk.PhotoImage(vista)
    fotos_tk[clave] = foto_tk
    lienzo.image = foto_tk
    lienzo.delete('all')
    lienzo.create_image(cx, cy, image=foto_tk)

    orig_h, orig_w = imagen_original.shape[:2]
    pix_orig = orig_w * orig_h
    pix_act = img_w * img_h

    if clave in ('directa', 'reducida'):
        pct = 100 * (1 - pix_act / pix_orig)
        info.config(text=f'{img_w} x {img_h} px  (-{pct:.2f}% píxeles)')
    else:
        info.config(text=f'{img_w} x {img_h} px')

    # Si es el panel original y hay un píxel seleccionado, dibujar marca discreta
    if clave == 'original' and pixel_seleccionado is not None:
        dibujar_marca_pixel(lienzo, pixel_seleccionado, geom)


def dibujar_marca_pixel(lienzo, pos, geom):
    """Dibuja una pequeña marca visual sobre el píxel seleccionado en la foto original."""
    fila, col = pos
    ox, oy = geom['ox'], geom['oy']
    factor = geom['factor']
    px = ox + int((col + 0.5) * factor)
    py = oy + int((fila + 0.5) * factor)
    lienzo.delete('marca_pixel')
    lienzo.create_oval(px - 4, py - 4, px + 4, py + 4, outline=ROJO, width=2, tag='marca_pixel')
    lienzo.create_line(px - 7, py, px + 7, py, fill=ROJO, width=1, tag='marca_pixel')
    lienzo.create_line(px, py - 7, px, py + 7, fill=ROJO, width=1, tag='marca_pixel')


def manejar_clic_original(event):
    """Detecta clic sobre la foto original, convierte a fila/columna y actualiza selección."""
    global pixel_seleccionado
    if imagen_original is None:
        return
    lienzo, _ = paneles.get('original', (None, None))
    if not lienzo:
        return
    orig_h, orig_w = imagen_original.shape[:2]
    res = convertir_clic_a_pixel(event.x, event.y, lienzo.winfo_width(), lienzo.winfo_height(), orig_w, orig_h, margen_interior=8)
    if res is None:
        return  # Clic fuera del área de la fotografía
    fila, col = res
    pixel_seleccionado = (fila, col)
    if lbl_pixel_info:
        lbl_pixel_info.config(text=f'Píxel seleccionado: fila {fila}, columna {col}')
    actualizar_panel('original')
    if callback_actualizar_calculo and ventana_calculo_actual and ventana_calculo_actual.winfo_exists():
        callback_actualizar_calculo(fila, col)


def mostrar_calculo():
    """Ventana interactiva Cálculo paso a paso — Convolución."""
    global ventana_calculo_actual, callback_actualizar_calculo
    if imagen_original is None or mascara_actual is None:
        messagebox.showinfo('Aviso', 'Cargue y procese una imagen primero.')
        return

    v = tk.Toplevel(raiz)
    v.title('Cálculo paso a paso — Convolución')
    v.geometry('920x600')
    v.minsize(780, 520)
    v.configure(bg=FONDO)
    ventana_calculo_actual = v

    top = tk.Frame(v, bg=BLANCO, padx=10, pady=6, relief='solid', bd=1)
    top.pack(fill='x', padx=8, pady=6)

    fila_inicial = pixel_seleccionado[0] if pixel_seleccionado else imagen_original.shape[0] // 2
    col_inicial = pixel_seleccionado[1] if pixel_seleccionado else imagen_original.shape[1] // 2
    f_var, c_var, ch_var = tk.IntVar(value=fila_inicial), tk.IntVar(value=col_inicial), tk.IntVar(value=0)

    vcmd_dec = (v.register(validar_decimal), '%P')
    vcmd_pix = (v.register(validar_pixel), '%P')
    vcmd_coord = (v.register(validar_entero_coord), '%P')

    tk.Label(top, text='Fila:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(side='left')
    sp_f = tk.Spinbox(top, from_=0, to=imagen_original.shape[0] - 1, textvariable=f_var, width=5,
                      command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord)
    sp_f.pack(side='left', padx=3)
    tk.Label(top, text='Col:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(side='left', padx=(6, 0))
    sp_c = tk.Spinbox(top, from_=0, to=imagen_original.shape[1] - 1, textvariable=c_var, width=5,
                      command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord)
    sp_c.pack(side='left', padx=3)

    for i, col, nom in ((0, '#dc2626', 'Rojo'), (1, '#16a34a', 'Verde'), (2, '#2563eb', 'Azul')):
        tk.Radiobutton(top, text=nom, variable=ch_var, value=i, bg=BLANCO, fg=col,
                       font=('Segoe UI', 9, 'bold'), command=lambda: cargar_vecindad()).pack(side='left', padx=3)

    tk.Button(top, text='Cargar píxel', command=lambda: cargar_vecindad(),
              bg='#263a55', fg=BLANCO, relief='flat', padx=8).pack(side='left', padx=6)
    tk.Button(top, text='Recalcular', command=lambda: recalcular(),
              bg=AZUL, fg=BLANCO, relief='flat', font=('Segoe UI', 9, 'bold'), padx=10).pack(side='right', padx=2)
    tk.Button(top, text='Aplicar máscara a la imagen', command=lambda: aplicar_mascara_desde_calculo(),
              bg=VERDE, fg=BLANCO, relief='flat', padx=8).pack(side='right', padx=2)

    body = tk.Frame(v, bg=BLANCO, padx=10, pady=8, relief='solid', bd=1)
    body.pack(fill='both', expand=True, padx=8, pady=4)

    # Indicador de relación con el salto (submuestreo)
    lbl_salto_rel = tk.Label(body, text='', font=('Segoe UI', 9, 'bold'), bg='#f8fafc',
                             relief='solid', bd=1, padx=6, pady=4)
    lbl_salto_rel.pack(fill='x', pady=(0, 6))

    tk.Label(body, text='Los valores editados se utilizan únicamente para practicar el cálculo manual.',
             font=('Segoe UI', 8, 'italic'), bg=BLANCO, fg=SECUNDARIO).pack(anchor='center', pady=(0, 4))

    m_row = tk.Frame(body, bg=BLANCO)
    m_row.pack(anchor='center', pady=4)

    cuadros = []
    for tit, col in (('1. Píxeles (I) [Editable]', TEXTO), ('2. Máscara (W)', AZUL), ('3. Productos (I x W)', VERDE)):
        g = tk.Frame(m_row, bg='#f8fafc', relief='solid', bd=1, padx=6, pady=6)
        g.pack(side='left', padx=6)
        tk.Label(g, text=tit, font=('Segoe UI', 8, 'bold'), bg='#f8fafc', fg=col).pack(pady=(0, 2))
        f_sub = tk.Frame(g, bg='#f8fafc')
        f_sub.pack()
        cuadros.append(f_sub)
        if len(cuadros) < 3:
            tk.Label(m_row, text='x' if len(cuadros) == 1 else '=', font=('Segoe UI', 16, 'bold'),
                     bg=BLANCO, fg=SECUNDARIO).pack(side='left', padx=2)

    lbl_sum = tk.Label(body, text='SUMA PONDERADA = -', font=('Segoe UI', 10, 'bold'), bg=BLANCO, fg=VERDE)
    lbl_sum.pack(pady=4)
    lbl_exp = tk.Label(body, text='', font=('Segoe UI', 8), bg=BLANCO, justify='center')
    lbl_exp.pack()

    tam = mascara_actual.shape[0]
    entradas_pix, entradas_pes, etiquetas_prod = [], [], []
    es_pers = (tipo_mascara.get() == 'Personalizada')

    for i in range(tam):
        f_pix, f_pes, f_prod = [], [], []
        for j in range(tam):
            ep = tk.Entry(cuadros[0], width=5, justify='center', font=('Consolas', 9), relief='solid', bd=1,
                          validate='key', validatecommand=vcmd_pix)
            ep.grid(row=i, column=j, padx=1, pady=1)
            ep.bind('<KeyRelease>', lambda ev: recalcular())
            ep.bind('<Return>', lambda ev: recalcular())
            f_pix.append(ep)

            # Matriz de pesos: editable solo en Personalizada
            ew = tk.Entry(cuadros[1], width=6, justify='center', font=('Consolas', 9), relief='solid', bd=1, fg=AZUL,
                          validate='key', validatecommand=vcmd_dec)
            ew.grid(row=i, column=j, padx=1, pady=1)
            if es_pers:
                ew.bind('<KeyRelease>', lambda ev: recalcular())
                ew.bind('<Return>', lambda ev: recalcular())
            else:
                ew.configure(state='readonly')
            f_pes.append(ew)

            lp = tk.Label(cuadros[2], text='0.0', width=6, font=('Consolas', 9), relief='solid', bd=1, bg=BLANCO, fg=VERDE)
            lp.grid(row=i, column=j, padx=1, pady=1)
            f_prod.append(lp)
        entradas_pix.append(f_pix)
        entradas_pes.append(f_pes)
        etiquetas_prod.append(f_prod)

    pesos_completos = np.zeros((tam, tam), dtype=np.float64)

    def cargar_vecindad():
        nonlocal pesos_completos
        f, c, ch = f_var.get(), c_var.get(), ch_var.get()
        if not (0 <= f < imagen_original.shape[0] and 0 <= c < imagen_original.shape[1]):
            return
        reg, pes, _, _ = explicar_pixel(imagen_original, mascara_actual, f, c)
        pesos_completos = pes.copy()
        for i in range(tam):
            for j in range(tam):
                entradas_pix[i][j].delete(0, 'end')
                entradas_pix[i][j].insert(0, str(int(reg[i, j, ch])))
                if not es_pers:
                    entradas_pes[i][j].configure(state='normal')
                entradas_pes[i][j].delete(0, 'end')
                entradas_pes[i][j].insert(0, f'{pes[i, j]:.4f}')
                if not es_pers:
                    entradas_pes[i][j].configure(state='readonly')
        recalcular()

    def recalcular():
        suma = 0.0
        for i in range(tam):
            for j in range(tam):
                try:
                    val_i = float(entradas_pix[i][j].get().strip().replace(',', '.'))
                except ValueError:
                    val_i = 0.0
                if es_pers:
                    try:
                        txt_w = entradas_pes[i][j].get().strip().replace(',', '.')
                        val_w = float(txt_w) if txt_w else 0.0
                    except ValueError:
                        val_w = 0.0
                else:
                    # En Promedio y Gaussiana usar la precisión completa float64 de NumPy
                    val_w = float(pesos_completos[i, j])

                prod = val_i * val_w
                suma += prod
                etiquetas_prod[i][j].config(text=f'{prod:.2f}')
        salida_clip = int(np.clip(np.rint(suma), 0, 255))
        f, c = f_var.get(), c_var.get()
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        muestreado = es_posicion_muestreada(f, c, sy, sx)

        if muestreado:
            lbl_salto_rel.config(
                text=f'[OK] La posición ({f}, {c}) SÍ se muestrea en la reducción con salto {sy}x{sx}. Forma parte de la imagen reducida.',
                fg=VERDE, bg='#f0fdf4'
            )
        else:
            lbl_salto_rel.config(
                text=f'[!] La posición ({f}, {c}) NO se muestrea con salto {sy}x{sx}. El cálculo es válido como demostración matemática.',
                fg='#b45309', bg='#fefce8'
            )

        lbl_sum.config(text=f'Suma ponderada (S) = {suma:.4f}  ->  Píxel final = {salida_clip}')
        lbl_exp.config(text='Procedimiento: cada intensidad I se multiplica por su peso W, la suma de los productos da el píxel resultante.')

    def aplicar_mascara_desde_calculo():
        if not es_pers:
            messagebox.showinfo('Aviso', 'La máscara activa es predefinida (Promedio o Gaussiana). Solo se pueden modificar los pesos de la máscara Personalizada.')
            return
        valores_w = []
        for i in range(tam):
            fila = []
            for j in range(tam):
                txt = entradas_pes[i][j].get().strip()
                if not txt:
                    messagebox.showerror('Pesos inválidos', f'La celda de peso ({i+1}, {j+1}) está vacía.')
                    return
                try:
                    val = float(txt.replace(',', '.'))
                except ValueError:
                    messagebox.showerror('Pesos inválidos', f'El valor "{txt}" en ({i+1}, {j+1}) no es un número válido.')
                    return
                if not np.isfinite(val) or val < 0:
                    messagebox.showerror('Pesos inválidos', f'El peso en ({i+1}, {j+1}) debe ser finito y no negativo.')
                    return
                fila.append(val)
            valores_w.append(fila)

        arr_w = np.asarray(valores_w, dtype=np.float64)
        if not np.any(arr_w > 0):
            messagebox.showerror('Pesos inválidos', 'La suma de los pesos debe ser mayor a cero (al menos un peso positivo).')
            return

        if not messagebox.askyesno('Confirmar aplicación', '¿Desea aplicar estos pesos como máscara personalizada global para toda la imagen?'):
            return

        matriz_original_w = arr_w[::-1, ::-1]
        establecer_matriz_codigo(matriz_original_w.tolist())
        procesar()
        messagebox.showinfo('Máscara aplicada', 'Los pesos modificados se normalizaron y se aplicaron a la imagen completa.')

    def actualizar_desde_externo(nueva_f, nueva_c):
        f_var.set(nueva_f)
        c_var.set(nueva_c)
        cargar_vecindad()

    callback_actualizar_calculo = actualizar_desde_externo
    cargar_vecindad()


def construir_interfaz():
    """Construye controles principales, panel lateral y cuadrícula de 4 vistas."""
    global raiz, nombre_imagen, estado, tipo_mascara, tamano_mascara
    global salto_horizontal, salto_vertical, paso_x_etiqueta, paso_y_etiqueta
    global lbl_pixel_info, marco_celdas, marco_herramientas_pers, lbl_integracion_texto, explicacion_mascara
    global lbl_res_orig, lbl_res_red, lbl_pix_tot, lbl_pct_red, lbl_cfg_masc, lbl_cfg_saltos, lbl_t_proc

    raiz = tk.Tk()
    raiz.title('Taller de IA — Suavizado y Reducción de Imágenes')
    raiz.geometry('1440x920')
    raiz.minsize(1040, 700)
    raiz.configure(bg=FONDO)

    # Cabecera principal
    cab = tk.Frame(raiz, bg='#14233b', height=65)
    cab.pack(fill='x')
    cab.pack_propagate(False)
    tit = tk.Frame(cab, bg='#14233b')
    tit.pack(side='left', padx=16, pady=6)
    tk.Label(tit, text='Suavizado y Reducción de Imágenes', font=('Segoe UI', 15, 'bold'), bg='#14233b', fg=BLANCO).pack(anchor='w')
    tk.Label(tit, text='Taller de IA — Universidad Técnica de Ambato | Convolución manual y saltos',
             font=('Segoe UI', 9), bg='#14233b', fg='#b6c7df').pack(anchor='w')

    btn_f = tk.Frame(cab, bg='#14233b')
    btn_f.pack(side='right', padx=14)
    for txt, cmd, col in (('Cargar imagen', cargar_imagen, AZUL), ('Procesar', procesar, AZUL),
                          ('Restaurar', restaurar, '#263a55'), ('Guardar PNG', guardar_png, VERDE)):
        tk.Button(btn_f, text=txt, command=cmd, font=('Segoe UI', 9, 'bold'), bg=col, fg=BLANCO,
                  bd=0, padx=12, pady=5, cursor='hand2').pack(side='left', padx=3)

    cuerpo = tk.Frame(raiz, bg=FONDO)
    cuerpo.pack(fill='both', expand=True, padx=8, pady=6)

    # Barra lateral de configuración
    sb = tk.Frame(cuerpo, bg=BLANCO, width=320, padx=10, pady=8, relief='solid', bd=1)
    sb.pack(side='left', fill='y', padx=(0, 8))
    sb.pack_propagate(False)

    nombre_imagen = tk.StringVar(value='Ninguna imagen')
    tk.Label(sb, text='Fotografía:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(anchor='w')
    tk.Label(sb, textvariable=nombre_imagen, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 9)).pack(anchor='w')

    lbl_pixel_info = tk.Label(sb, text='Píxel: haga clic en la foto original', font=('Segoe UI', 8, 'italic'),
                              fg=AZUL, bg=BLANCO)
    lbl_pixel_info.pack(anchor='w', pady=(0, 6))

    # Sección 1: Máscara
    tk.Label(sb, text='1. Máscara de suavizado', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w')
    tipo_mascara, tamano_mascara = tk.StringVar(value='Gaussiana'), tk.StringVar(value='3')
    cb_t = ttk.Combobox(sb, textvariable=tipo_mascara, values=('Gaussiana', 'Promedio', 'Personalizada'), state='readonly')
    cb_t.pack(fill='x', pady=2)
    cb_s = ttk.Combobox(sb, textvariable=tamano_mascara, values=('3', '5', '7'), state='readonly')
    cb_s.pack(fill='x', pady=2)

    c_box = tk.Frame(sb, bg='#f8fafc', relief='solid', bd=1, padx=4, pady=4)
    c_box.pack(fill='x', pady=3)
    marco_celdas = tk.Frame(c_box, bg='#f8fafc')
    marco_celdas.pack(anchor='center')

    marco_herramientas_pers = tk.Frame(c_box, bg='#f8fafc')
    pr_f = tk.Frame(marco_herramientas_pers, bg='#f8fafc')
    pr_f.pack(fill='x', pady=(2, 3))
    tk.Button(pr_f, text='1s', command=lambda: aplicar_preset('unos'), font=('Segoe UI', 7),
              relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    tk.Button(pr_f, text='1-2-1', command=lambda: aplicar_preset('gauss'), font=('Segoe UI', 7),
              relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    tk.Button(pr_f, text='Centro 5', command=lambda: aplicar_preset('centro'), font=('Segoe UI', 7),
              relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    lbl_integracion_texto = tk.Label(marco_herramientas_pers, text='Integrada: -', font=('Segoe UI', 7),
                                     fg=AZUL, bg='#f8fafc', wraplength=280, justify='left')
    lbl_integracion_texto.pack(anchor='w', pady=1)

    explicacion_mascara = tk.StringVar()
    tk.Label(sb, textvariable=explicacion_mascara, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 8),
             wraplength=290).pack(anchor='w', pady=1)

    # Sección 2: Saltos de reducción
    tk.Label(sb, text='2. Saltos de reducción', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w', pady=(4, 0))
    salto_horizontal, salto_vertical = tk.IntVar(value=2), tk.IntVar(value=2)
    paso_x_etiqueta, paso_y_etiqueta = tk.StringVar(value='2'), tk.StringVar(value='2')

    for nom, var, lbl in (('Salto horizontal:', salto_horizontal, paso_x_etiqueta),
                          ('Salto vertical:', salto_vertical, paso_y_etiqueta)):
        f = tk.Frame(sb, bg=BLANCO)
        f.pack(fill='x', pady=(1, 0))
        tk.Label(f, text=nom, bg=BLANCO, font=('Segoe UI', 8)).pack(side='left')
        tk.Label(f, textvariable=lbl, bg=BLANCO, font=('Segoe UI', 9, 'bold'), fg=AZUL).pack(side='right')
        sc = tk.Scale(sb, from_=1, to=8, orient='horizontal', showvalue=False, variable=var,
                      command=actualizar_reduccion, bg=BLANCO, bd=0, highlightthickness=0)
        sc.pack(fill='x')

    # Sección 3: Resultados del procesamiento (resumen compacto)
    card_res = tk.LabelFrame(sb, text='Resultados del procesamiento', font=('Segoe UI', 9, 'bold'),
                             bg='#f8fafc', fg='#1e293b', padx=8, pady=4, relief='solid', bd=1)
    card_res.pack(fill='x', pady=(4, 0))

    lbl_res_orig = tk.Label(card_res, text='Original: -', font=('Segoe UI', 8), bg='#f8fafc', anchor='w')
    lbl_res_orig.pack(fill='x')
    lbl_res_red = tk.Label(card_res, text='Reducida: -', font=('Segoe UI', 8), bg='#f8fafc', anchor='w')
    lbl_res_red.pack(fill='x')
    lbl_pix_tot = tk.Label(card_res, text='Píxeles: -', font=('Segoe UI', 8), bg='#f8fafc', anchor='w')
    lbl_pix_tot.pack(fill='x')
    lbl_pct_red = tk.Label(card_res, text='Reducción: -', font=('Segoe UI', 8, 'bold'), bg='#f8fafc', fg=VERDE, anchor='w')
    lbl_pct_red.pack(fill='x')
    lbl_cfg_masc = tk.Label(card_res, text='Máscara: -', font=('Segoe UI', 8), bg='#f8fafc', anchor='w')
    lbl_cfg_masc.pack(fill='x')
    lbl_cfg_saltos = tk.Label(card_res, text='Saltos: -', font=('Segoe UI', 8), bg='#f8fafc', anchor='w')
    lbl_cfg_saltos.pack(fill='x')
    lbl_t_proc = tk.Label(card_res, text='Tiempo cálculo: -', font=('Segoe UI', 8), bg='#f8fafc', fg=AZUL, anchor='w')
    lbl_t_proc.pack(fill='x')

    tk.Button(sb, text='Ver cálculo paso a paso', command=mostrar_calculo,
              font=('Segoe UI', 9, 'bold'), bg='#e6edff', fg=AZUL, relief='flat', padx=6, pady=5).pack(fill='x', pady=(6, 0))

    # Mosaico de 4 vistas
    mos = tk.Frame(cuerpo, bg=FONDO)
    mos.pack(side='left', fill='both', expand=True)
    mos.grid_columnconfigure(0, weight=1, uniform='img')
    mos.grid_columnconfigure(1, weight=1, uniform='img')
    mos.grid_rowconfigure(0, weight=1)
    mos.grid_rowconfigure(1, weight=1)

    vistas = [('original', 'IMAGEN ORIGINAL (Clic para seleccionar píxel)'),
              ('suavizada', 'IMAGEN SUAVIZADA'),
              ('directa', 'REDUCCIÓN DIRECTA'),
              ('reducida', 'REDUCCIÓN CON SUAVIZADO')]

    for idx, (k, tit_txt) in enumerate(vistas):
        m = tk.Frame(mos, bg=BLANCO, relief='solid', bd=1, padx=3, pady=3)
        m.grid(row=idx // 2, column=idx % 2, sticky='nsew', padx=3, pady=3)
        m.grid_rowconfigure(1, weight=1)
        m.grid_columnconfigure(0, weight=1)

        tk.Label(m, text=tit_txt, font=('Segoe UI', 9, 'bold'), bg=BLANCO,
                 fg=AZUL if k == 'reducida' else TEXTO).grid(row=0, column=0, pady=1)

        lz = tk.Canvas(m, bg=FONDO_IMG, highlightthickness=0, cursor='crosshair' if k == 'original' else '')
        lz.grid(row=1, column=0, sticky='nsew')
        inf = tk.Label(m, text='-', font=('Segoe UI', 8), bg=BLANCO, fg=SECUNDARIO)
        inf.grid(row=2, column=0, pady=1)
        paneles[k] = (lz, inf)
        lz.bind('<Configure>', lambda e, c=k: actualizar_panel(c))

        if k == 'original':
            lz.bind('<Button-1>', manejar_clic_original)

    # Pie de estado
    pie = tk.Frame(raiz, bg=FONDO, padx=10, pady=2)
    pie.pack(fill='x')
    estado = tk.StringVar(value='Listo.')
    tk.Label(pie, textvariable=estado, bg=FONDO, fg=SECUNDARIO, font=('Segoe UI', 8)).pack(anchor='w')

    cb_t.bind('<<ComboboxSelected>>', actualizar_mascara)
    cb_s.bind('<<ComboboxSelected>>', actualizar_mascara)
    actualizar_mascara()
    return raiz


if __name__ == '__main__':
    construir_interfaz()
    for p in ('imagenes/paisaje.png', 'imagenes/paisaje.jpg'):
        if Path(p).exists():
            cargar_imagen(p)
            break
    raiz.mainloop()
