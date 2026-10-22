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
                            convertir_clic_a_pixel, es_posicion_muestreada,
                            aplicar_ruido, aplicar_filtro_moda)

# Configuración visual y paleta moderna
FONDO, BLANCO, TEXTO, SECUNDARIO = '#eef2f7', '#ffffff', '#18243a', '#64748b'
AZUL, VERDE, FONDO_IMG, ROJO = '#2458dc', '#059669', '#141c29', '#dc2626'

# Matriz personalizada inicial (Gaussiana binomial 3x3)
MATRIZ_PERSONALIZADA_DEFECTO = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
matriz_personalizada_guardada = [list(f) for f in MATRIZ_PERSONALIZADA_DEFECTO]

# Estado global de imágenes y geometría
raiz = None
imagen_base = imagen_original = imagen_reducida = imagen_directa = None
mascara_actual = datos_actuales = None
ruta_actual = ''
pixel_seleccionado = None

# Variables de control de la interfaz
nombre_imagen = tipo_mascara = tamano_mascara = None
salto_horizontal = salto_vertical = paso_x_etiqueta = paso_y_etiqueta = None
tipo_ruido = intensidad_ruido = etiqueta_intensidad_ruido = None
estado = explicacion_mascara = lbl_pixel_info = None
marco_celdas = marco_herramientas_pers = None
paneles, fotos_tk, celdas_matriz = {}, {}, []

# Etiquetas de la tarjeta de métricas
lbl_res_orig = lbl_res_red = lbl_pix_tot = lbl_pct_red = None
lbl_cfg_masc = lbl_cfg_saltos = lbl_t_proc = None

# Control y sincronización de la ventana de cálculo interactivo
ventana_calculo_actual = None
callback_actualizar_calculo = None
callback_actualizar_salto = None
_aplicando_desde_calculo = False

def cerrar_calculo_seguro():
    """Cierra la ventana de cálculo paso a paso y limpia callbacks y referencias."""
    global ventana_calculo_actual, callback_actualizar_calculo, callback_actualizar_salto
    if ventana_calculo_actual is not None:
        try:
            if ventana_calculo_actual.winfo_exists():
                ventana_calculo_actual.destroy()
        except Exception:
            pass
    ventana_calculo_actual = None
    callback_actualizar_calculo = callback_actualizar_salto = None

def validar_decimal(texto):
    """Permite exclusivamente dígitos y como máximo un punto o coma decimal."""
    return texto == '' or (texto in ('.', ',') or (
        all(c in '0123456789.,' for c in texto) and texto.count('.') + texto.count(',') <= 1
    ))

def validar_pixel(texto):
    """Permite exclusivamente enteros de 0 a 255 para intensidades RGB."""
    return texto == '' or (texto.isdigit() and 0 <= int(texto) <= 255)

def validar_entero_coord(texto):
    """Permite exclusivamente enteros no negativos para coordenadas de fila/columna."""
    return texto == '' or texto.isdigit()

def establecer_matriz_codigo(nueva_matriz):
    """Define o cambia la matriz personalizada directamente desde código."""
    global matriz_personalizada_guardada
    matriz_personalizada_guardada = [list(f) for f in nueva_matriz]
    if tipo_mascara and tipo_mascara.get() == 'Personalizada':
        actualizar_mascara()

def cargar_imagen(ruta=None):
    """Carga imagen RGB sin alterar su resolución real y la procesa."""
    global imagen_base, imagen_original, ruta_actual, pixel_seleccionado
    if not ruta:
        ruta = filedialog.askopenfilename(
            title='Seleccionar imagen',
            filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp')]
        )
    if not ruta:
        return
    try:
        with Image.open(ruta) as f:
            imagen_base = np.array(f.convert('RGB'), dtype=np.uint8)
        imagen_original = aplicar_ruido_activo()
        ruta_actual = str(ruta)
        pixel_seleccionado = (imagen_original.shape[0] // 2, imagen_original.shape[1] // 2)
        if nombre_imagen:
            nombre_imagen.set(Path(ruta).name)
        if lbl_pixel_info:
            lbl_pixel_info.config(text=f'Píxel seleccionado: fila {pixel_seleccionado[0]}, columna {pixel_seleccionado[1]}')
        reabrir_calculo = ventana_calculo_actual and ventana_calculo_actual.winfo_exists()
        if reabrir_calculo:
            cerrar_calculo_seguro()
        procesar()
        if reabrir_calculo and imagen_original is not None:
            mostrar_calculo()
    except Exception as e:
        messagebox.showerror('Error al cargar imagen', str(e))

def aplicar_ruido_activo(*args):
    """Genera la entrada ruidosa desde la imagen cargada, sin degradarla acumulativamente."""
    global imagen_original
    if imagen_base is None:
        return None
    tipo = tipo_ruido.get() if tipo_ruido else 'Sin ruido'
    intensidad = float(intensidad_ruido.get()) if intensidad_ruido else 0.0
    imagen_original = aplicar_ruido(imagen_base, tipo, intensidad, semilla=2026)
    if etiqueta_intensidad_ruido:
        etiqueta_intensidad_ruido.set(f'{intensidad:.2f}')
    procesar()
    return imagen_original

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
                h, w = imagen_reducida.shape[:2]
                estado.set(f'Guardada con éxito: {Path(ruta).name} ({w}x{h} px)')
        except Exception as e:
            messagebox.showerror('Error al guardar', str(e))

def restaurar():
    """Restaura la fotografía original y los parámetros predeterminados."""
    global pixel_seleccionado
    if imagen_original is None:
        return
    tipo_mascara.set('Gaussiana')
    tamano_mascara.set('3')
    tipo_ruido.set('Sin ruido')
    intensidad_ruido.set(0.0)
    salto_horizontal.set(2)
    salto_vertical.set(2)
    paso_x_etiqueta.set('2')
    paso_y_etiqueta.set('2')
    pixel_seleccionado = (imagen_original.shape[0] // 2, imagen_original.shape[1] // 2)
    if lbl_pixel_info:
        lbl_pixel_info.config(text=f'Píxel seleccionado: fila {pixel_seleccionado[0]}, columna {pixel_seleccionado[1]}')
    aplicar_ruido_activo()
    actualizar_mascara()
    if estado:
        estado.set('Configuración inicial restaurada (Gaussiana 3x3, salto 2x2).')

def mascara_desde_interfaz():
    """Obtiene máscara normalizada según las opciones activas."""
    global matriz_personalizada_guardada
    tipo, tam = tipo_mascara.get(), int(tamano_mascara.get())
    if tipo == 'Moda':
        return None
    if tipo != 'Personalizada':
        return crear_mascara(tipo, tam)
    valores = []
    for f in range(tam):
        fila = []
        for c in range(tam):
            if f >= len(celdas_matriz) or c >= len(celdas_matriz[f]):
                raise ValueError('La cuadrícula no coincide con el tamaño seleccionado.')
            txt = celdas_matriz[f][c].get().strip().replace(',', '.')
            if not txt:
                raise ValueError(f'La celda (fila {f+1}, columna {c+1}) está vacía. Ingrese un valor numérico.')
            try:
                num = float(txt)
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
    if preset == 'gauss':
        v = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [1, 6, 15, 20, 15, 6, 1]}.get(tam, [1] * tam)
        mat = np.outer(v, v)
    elif preset == 'centro':
        mat = np.ones((tam, tam), dtype=int)
        mat[tam // 2, tam // 2] = 5
    else:  # 'unos'
        mat = np.ones((tam, tam), dtype=int)
    for f in range(tam):
        for c in range(tam):
            celdas_matriz[f][c].delete(0, 'end')
            celdas_matriz[f][c].insert(0, str(int(mat[f, c])))
    procesar()
    if (ventana_calculo_actual is not None
            and ventana_calculo_actual.winfo_exists()
            and callback_actualizar_calculo is not None):
        callback_actualizar_calculo(pixel_seleccionado[0], pixel_seleccionado[1])

def actualizar_mascara(*args):
    """Reconstruye casillas de la máscara: editables en Personalizada, solo lectura en Promedio y Gaussiana."""
    global celdas_matriz
    if not marco_celdas:
        return
    tipo, tam = tipo_mascara.get(), int(tamano_mascara.get())
    calculo_estaba_abierto = (
        ventana_calculo_actual is not None
        and ventana_calculo_actual.winfo_exists()
        and not _aplicando_desde_calculo
    )
    if calculo_estaba_abierto:
        cerrar_calculo_seguro()
    for w in marco_celdas.winfo_children():
        w.destroy()
    celdas_matriz = []
    vcmd_dec = (raiz.register(validar_decimal), '%P') if raiz else None
    # Configuración de dimensiones y fuentes según el tamaño de la matriz
    if tam == 3:
        px, py, w_col_pers, f_pers = 2, 2, 5, ('Consolas', 10, 'bold')
        w_col_ro, f_ro, fmt = 6, ('Consolas', 9), lambda v: f'{v:.4f}'
    elif tam == 5:
        px, py, w_col_pers, f_pers = 1, 1, 4, ('Consolas', 9, 'bold')
        w_col_ro, f_ro, fmt = 5, ('Consolas', 8), lambda v: f'{v:.3f}'
    else:  # 7x7
        px, py, w_col_pers, f_pers = 1, 1, 5, ('Consolas', 8, 'bold')
        w_col_ro, f_ro, fmt = 5, ('Consolas', 7), lambda v: f'{v:.3f}'
    if tipo == 'Moda':
        for w in marco_celdas.winfo_children():
            w.destroy()
        celdas_matriz = []
        marco_herramientas_pers.pack_forget()
        explicacion_mascara.set('Moda: conserva el valor más frecuente de cada vecindad RGB.')
        procesar()
        return
    personalizada = tipo == 'Personalizada'
    m = None if personalizada else crear_mascara(tipo, tam)
    ancho = w_col_pers if personalizada else w_col_ro
    fuente = f_pers if personalizada else f_ro

    # Una sola cuadrícula para los tres filtros: cambia únicamente su contenido y edición.
    for f in range(tam):
        fila_w = []
        for c in range(tam):
            opciones = {'validate': 'key', 'validatecommand': vcmd_dec} if personalizada else {}
            e = tk.Entry(marco_celdas, width=ancho, justify='center', font=fuente,
                         relief='solid', bd=1, **opciones)
            e.grid(row=f, column=c, padx=px, pady=py)
            if personalizada:
                existe = f < len(matriz_personalizada_guardada) and c < len(matriz_personalizada_guardada[f])
                valor = str(matriz_personalizada_guardada[f][c]) if existe else '1'
                e.bind('<Return>', lambda evento: procesar())
            else:
                valor = fmt(m[f, c])
            e.insert(0, valor)
            if not personalizada:
                e.configure(state='readonly', readonlybackground='#f1f5f9', fg=SECUNDARIO)
            fila_w.append(e)
        celdas_matriz.append(fila_w)

    if personalizada:
        marco_herramientas_pers.pack(fill='x', pady=2)
        explicacion_mascara.set('Personalizada: edite los pesos y presione "Procesar".')
    else:
        marco_herramientas_pers.pack_forget()
        explicacion_mascara.set(f'Pesos normalizados ({tipo}): suma = {m.sum():.4f}')
    procesar()
    if calculo_estaba_abierto and imagen_original is not None:
        mostrar_calculo()

def actualizar_reduccion(*args):
    """Calcula y visualiza la reducción con saltos en tiempo real y actualiza el resumen."""
    global imagen_reducida, imagen_directa, datos_actuales
    if imagen_original is None:
        return
    sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
    paso_x_etiqueta.set(str(sx))
    paso_y_etiqueta.set(str(sy))
    t0 = time.perf_counter()
    if tipo_mascara.get() == 'Moda':
        imagen_reducida = aplicar_filtro_moda(imagen_original, int(tamano_mascara.get()), sy, sx)
    else:
        masc = mascara_actual if mascara_actual is not None else mascara_desde_interfaz()
        imagen_reducida = aplicar_convolucion(imagen_original, masc, sy, sx)
    imagen_directa = reducir_sin_suavizado(imagen_original, sy, sx)
    t_ms = (time.perf_counter() - t0) * 1000
    datos_actuales = calcular_datos(imagen_original, imagen_reducida, t_ms)
    actualizar_panel('directa')
    actualizar_panel('reducida')
    actualizar_resumen_resultados(datos_actuales, sx, sy)
    if estado:
        h, w = imagen_reducida.shape[:2]
        pct = datos_actuales['reduccion']
        estado.set(f'[OK] Salto {sx}x{sy} -> {w}x{h} px | Reducción: {pct:.2f}% píxeles ({t_ms:.1f} ms)')
    if callback_actualizar_salto and ventana_calculo_actual and ventana_calculo_actual.winfo_exists():
        callback_actualizar_salto()

def actualizar_resumen_resultados(datos, sx, sy):
    """Actualiza la sección Resultados del Procesamiento en la barra lateral."""
    if not lbl_res_orig or not datos:
        return
    lbl_res_orig.config(text=f'Original: {datos["ancho_original"]} x {datos["alto_original"]} px')
    lbl_res_red.config(text=f'Reducida: {datos["ancho_final"]} x {datos["alto_final"]} px')
    lbl_pix_tot.config(text=f'Píxeles: {datos["pixeles_original"]:,} -> {datos["pixeles_final"]:,}')
    lbl_pct_red.config(text=f'Reducción: {datos["reduccion"]:.2f} %')
    lbl_cfg_masc.config(text=f'Máscara: {tipo_mascara.get()} ({tamano_mascara.get()}x{tamano_mascara.get()})')
    lbl_cfg_saltos.config(text=f'Saltos: H: {sx}  |  V: {sy}')
    lbl_t_proc.config(text=f'Tiempo cálculo: {datos["tiempo_ms"]:.1f} ms')

def procesar():
    """Punto único de procesamiento: reducción con saltos, reducción directa y CSV."""
    global mascara_actual
    if imagen_original is None:
        if estado:
            estado.set('Cargue una fotografía para comenzar.')
        return
    try:
        mascara_actual = mascara_desde_interfaz()
        actualizar_panel('original')
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
    try:
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        exportar_metricas_csv('resultados_procesamiento.csv', Path(ruta_actual).name,
                              datos_actuales, tipo_mascara.get(), int(tamano_mascara.get()), sx, sy)
    except Exception:
        pass

def actualizar_panel(clave):
    """Muestra la imagen adaptada al panel manteniendo la relación de aspecto."""
    img = {'original': imagen_original, 'directa': imagen_directa,
           'reducida': imagen_reducida}.get(clave)
    if img is None or imagen_original is None or clave not in paneles:
        return
    lienzo, info = paneles[clave]
    img_h, img_w = img.shape[:2]
    geom = calcular_geometria_panel(lienzo.winfo_width(), lienzo.winfo_height(), img_w, img_h, margen_interior=8)
    if geom is None:
        return
    foto = Image.fromarray(img)
    vista = foto.resize((geom['vw'], geom['vh']), Image.Resampling.LANCZOS)
    foto_tk = ImageTk.PhotoImage(vista)
    fotos_tk[clave] = foto_tk
    lienzo.image = foto_tk
    lienzo.delete('all')
    lienzo.create_image(geom['cx'], geom['cy'], image=foto_tk)
    orig_h, orig_w = imagen_original.shape[:2]
    if clave in ('directa', 'reducida'):
        pct = 100 * (1 - (img_w * img_h) / (orig_w * orig_h))
        info.config(text=f'{img_w} x {img_h} px  (-{pct:.2f}% píxeles)')
    else:
        info.config(text=f'{img_w} x {img_h} px')
    if clave == 'original' and pixel_seleccionado is not None:
        dibujar_marca_pixel(lienzo, pixel_seleccionado, geom)

def dibujar_marca_pixel(lienzo, pos, geom):
    """Dibuja una marca visual en forma de mira sobre el píxel seleccionado en la foto original."""
    fila, col = pos
    orig_h, orig_w = imagen_original.shape[:2]
    px = geom['ox'] + int((col + 0.5) * geom['vw'] / orig_w)
    py = geom['oy'] + int((fila + 0.5) * geom['vh'] / orig_h)
    lienzo.delete('marca_pixel')
    lienzo.create_oval(px - 4, py - 4, px + 4, py + 4, outline=ROJO, width=2, tag='marca_pixel')
    lienzo.create_line(px - 7, py, px + 7, py, fill=ROJO, width=1, tag='marca_pixel')
    lienzo.create_line(px, py - 7, px, py + 7, fill=ROJO, width=1, tag='marca_pixel')

def manejar_clic_original(event):
    """Convierte el clic sobre la foto original a coordenadas reales de píxel."""
    global pixel_seleccionado
    if imagen_original is None:
        return
    lienzo, _ = paneles.get('original', (None, None))
    if not lienzo:
        return
    orig_h, orig_w = imagen_original.shape[:2]
    res = convertir_clic_a_pixel(event.x, event.y, lienzo.winfo_width(), lienzo.winfo_height(), orig_w, orig_h, margen_interior=8)
    if res is None:
        return
    fila, col = res
    pixel_seleccionado = (fila, col)
    if lbl_pixel_info:
        lbl_pixel_info.config(text=f'Píxel seleccionado: fila {fila}, columna {col}')
    actualizar_panel('original')
    if callback_actualizar_calculo and ventana_calculo_actual and ventana_calculo_actual.winfo_exists():
        callback_actualizar_calculo(fila, col)

def mostrar_calculo():
    """Ventana interactiva Cálculo paso a paso — Convolución."""
    global ventana_calculo_actual, callback_actualizar_calculo, callback_actualizar_salto
    if tipo_mascara.get() == 'Moda':
        messagebox.showinfo('Filtro de moda', 'El cálculo paso a paso muestra convoluciones. Seleccione Promedio, Gaussiana o Personalizada para usarlo.')
        return
    if imagen_original is None or mascara_actual is None:
        messagebox.showinfo('Aviso', 'Cargue y procese una imagen primero.')
        return
    if ventana_calculo_actual and ventana_calculo_actual.winfo_exists():
        ventana_calculo_actual.lift()
        ventana_calculo_actual.focus_force()
        return
    tam = mascara_actual.shape[0]
    es_pers = (tipo_mascara.get() == 'Personalizada')
    # Dimensiones, fuentes y anchos de celdas según tamaño
    if tam == 7:
        dims, min_dims = (1240, 450), (980, 420)
        w_pix, w_pes, w_prod = 4, 5, 5
        f_mat = ('Consolas', 9)
        alto_canvas = 218
    elif tam == 5:
        dims, min_dims = (1120, 395), (900, 375)
        w_pix, w_pes, w_prod = 5, 6, 6
        f_mat = ('Consolas', 9)
        alto_canvas = 165
    else:  # 3x3
        dims, min_dims = (960, 355), (800, 340)
        w_pix, w_pes, w_prod = 5, 7, 7
        f_mat = ('Consolas', 10)
        alto_canvas = 120

    scr_w = raiz.winfo_screenwidth() if raiz else 1366
    scr_h = raiz.winfo_screenheight() if raiz else 768
    v_w = min(dims[0], max(scr_w - 40, 780))
    v_h = min(dims[1], max(scr_h - 80, 320))
    v = tk.Toplevel(raiz)
    v.title(f'Cálculo paso a paso — Convolución ({tam}x{tam})')
    v.geometry(f'{v_w}x{v_h}')
    v.minsize(*min_dims)
    v.configure(bg=FONDO)
    v.protocol("WM_DELETE_WINDOW", cerrar_calculo_seguro)
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
    tk.Spinbox(top, from_=0, to=imagen_original.shape[0] - 1, textvariable=f_var, width=5,
               command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord).pack(side='left', padx=3)
    tk.Label(top, text='Col:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(side='left', padx=(6, 0))
    tk.Spinbox(top, from_=0, to=imagen_original.shape[1] - 1, textvariable=c_var, width=5,
               command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord).pack(side='left', padx=3)
    for i, col, nom in ((0, ROJO, 'Rojo'), (1, VERDE, 'Verde'), (2, AZUL, 'Azul')):
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
    lbl_salto_rel = tk.Label(body, text='', font=('Segoe UI', 9, 'bold'), bg='#f8fafc',
                             relief='solid', bd=1, padx=6, pady=4)
    lbl_salto_rel.pack(fill='x', pady=(0, 4))
    tk.Label(body, text='Los valores editados se utilizan únicamente para practicar el cálculo manual.',
             font=('Segoe UI', 8, 'italic'), bg=BLANCO, fg=SECUNDARIO).pack(anchor='center', pady=(0, 2))
    contenedor_canvas = tk.Frame(body, bg=BLANCO)
    contenedor_canvas.pack(fill='x', pady=2)
    canvas_mat = tk.Canvas(contenedor_canvas, bg=BLANCO, highlightthickness=0,
                           height=alto_canvas)
    h_scroll = ttk.Scrollbar(contenedor_canvas, orient='horizontal', command=canvas_mat.xview)
    canvas_mat.configure(xscrollcommand=h_scroll.set)
    m_row = tk.Frame(canvas_mat, bg=BLANCO)
    canvas_win = canvas_mat.create_window((0, 0), window=m_row, anchor='nw')
    def ajustar_scroll_matrices(ev=None):
        canvas_mat.update_idletasks()
        bbox = canvas_mat.bbox('all')
        if bbox:
            canvas_mat.configure(scrollregion=bbox)
            req_w, can_w = m_row.winfo_reqwidth(), canvas_mat.winfo_width()
            if req_w < can_w:
                canvas_mat.coords(canvas_win, (can_w - req_w) // 2, 0)
                h_scroll.pack_forget()
            else:
                canvas_mat.coords(canvas_win, 0, 0)
                h_scroll.pack(fill='x', side='bottom')
    m_row.bind('<Configure>', ajustar_scroll_matrices)
    canvas_mat.bind('<Configure>', ajustar_scroll_matrices)
    canvas_mat.pack(fill='x')
    cuadros = []
    titulos = [
        ('1. Píxeles (I) [Editable]', TEXTO),
        ('2. Máscara (W) [Personalizada]' if es_pers else '2. Máscara (W) [Normalizada]', AZUL),
        ('3. Productos (I x W_conv)', VERDE)
    ]
    for tit, col in titulos:
        g = tk.Frame(m_row, bg='#f8fafc', relief='solid', bd=1, padx=4, pady=4)
        g.pack(side='left', padx=4)
        tk.Label(g, text=tit, font=('Segoe UI', 8, 'bold'), bg='#f8fafc', fg=col).pack(pady=(0, 2))
        f_sub = tk.Frame(g, bg='#f8fafc')
        f_sub.pack()
        cuadros.append(f_sub)
        if len(cuadros) < 3:
            tk.Label(m_row, text='x' if len(cuadros) == 1 else '=', font=('Segoe UI', 14, 'bold'),
                     bg=BLANCO, fg=SECUNDARIO).pack(side='left', padx=2)
    lbl_sum = tk.Label(body, text='SUMA PONDERADA = -', font=('Segoe UI', 10, 'bold'), bg=BLANCO, fg=VERDE)
    lbl_sum.pack(pady=2)
    lbl_pesos_info = tk.Label(body, text='', font=('Segoe UI', 8, 'bold'), bg=BLANCO, fg=AZUL)
    lbl_pesos_info.pack(pady=1)
    lbl_exp = tk.Label(body, text='', font=('Segoe UI', 8), bg=BLANCO, justify='center')
    lbl_exp.pack(pady=1)
    entradas_pix, entradas_pes, etiquetas_prod = [], [], []
    for i in range(tam):
        f_pix, f_pes, f_prod = [], [], []
        for j in range(tam):
            ep = tk.Entry(cuadros[0], width=w_pix, justify='center', font=f_mat, relief='solid', bd=1,
                          validate='key', validatecommand=vcmd_pix)
            ep.grid(row=i, column=j, padx=1, pady=1)
            ep.bind('<KeyRelease>', lambda ev: recalcular())
            ep.bind('<Return>', lambda ev: recalcular())
            f_pix.append(ep)
            ew = tk.Entry(cuadros[1], width=w_pes, justify='center', font=f_mat, relief='solid', bd=1, fg=AZUL,
                          validate='key', validatecommand=vcmd_dec)
            ew.grid(row=i, column=j, padx=1, pady=1)
            if es_pers:
                ew.bind('<KeyRelease>', lambda ev: recalcular())
                ew.bind('<Return>', lambda ev: recalcular())
            else:
                ew.configure(state='readonly')
            f_pes.append(ew)
            lp = tk.Label(cuadros[2], text='0.0', width=w_prod, font=f_mat, relief='solid', bd=1, bg=BLANCO, fg=VERDE)
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
                if es_pers:
                    entradas_pes[i][j].delete(0, 'end')
                    if i < len(matriz_personalizada_guardada) and j < len(matriz_personalizada_guardada[i]):
                        v_peso = matriz_personalizada_guardada[i][j]
                    else:
                        v_peso = 1.0
                    entradas_pes[i][j].insert(0, str(v_peso))
                else:
                    entradas_pes[i][j].configure(state='normal')
                    entradas_pes[i][j].delete(0, 'end')
                    entradas_pes[i][j].insert(0, f'{pes[i, j]:.4f}')
                    entradas_pes[i][j].configure(state='readonly')
        recalcular()
    def leer_matriz(entradas, es_pixel=True):
        mat = np.zeros((tam, tam), dtype=np.float64)
        for i in range(tam):
            for j in range(tam):
                txt = entradas[i][j].get().strip().replace(',', '.')
                if not txt:
                    tipo_desc = 'las intensidades de los píxeles' if es_pixel else 'los pesos de la máscara'
                    return None, f'[Edición pendiente: complete {tipo_desc}]'
                try:
                    val = float(txt)
                except ValueError:
                    pref = 'Píxel' if es_pixel else 'Peso no numérico'
                    return None, f'[{pref} "{txt}" en celda ({i+1}, {j+1})]' if not es_pixel else f'[Valor inválido "{txt}" en píxel ({i+1}, {j+1})]'
                if es_pixel and not (0 <= val <= 255):
                    return None, f'[Píxel ({i+1}, {j+1}) fuera de rango 0-255]'
                if not es_pixel and (not np.isfinite(val) or val < 0):
                    return None, f'[Peso inválido ({val}) en celda ({i+1}, {j+1}): debe ser finito y no negativo]'
                mat[i, j] = val
        return mat, None
    def recalcular():
        f, c = f_var.get(), c_var.get()
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        actualizar_indicador_salto()
        # 1. Validación de intensidades de píxeles
        matriz_i, err_i = leer_matriz(entradas_pix, es_pixel=True)
        if err_i:
            lbl_sum.config(text=err_i, fg='#b45309' if 'pendiente' in err_i else ROJO)
            if 'pendiente' in err_i:
                lbl_exp.config(text='Ingrese números de 0 a 255 para cada celda de la vecindad.')
            return
        # 2. Validación y normalización de la máscara
        if es_pers:
            matriz_w_cruda, err_w = leer_matriz(entradas_pes, es_pixel=False)
            if err_w:
                lbl_sum.config(text=err_w, fg='#b45309' if 'pendiente' in err_w else ROJO)
                if 'pendiente' in err_w:
                    lbl_exp.config(text='Ingrese números finitos no negativos en las celdas de la máscara.')
                for fi in range(tam):
                    for cj in range(tam):
                        etiquetas_prod[fi][cj].config(text='-')
                return
            suma_cruda = float(matriz_w_cruda.sum())
            if suma_cruda <= 0 or not np.any(matriz_w_cruda > 0):
                lbl_sum.config(text='[Error: La suma de los pesos debe ser mayor a 0 (al menos un peso positivo)]', fg=ROJO)
                lbl_pesos_info.config(text='Suma de pesos crudos = 0.0 | No es posible normalizar')
                for fi in range(tam):
                    for cj in range(tam):
                        etiquetas_prod[fi][cj].config(text='-')
                return
            masc_norm = crear_mascara('Personalizada', tam, matriz_w_cruda)
            lbl_pesos_info.config(text=f'Pesos: suma original = {suma_cruda:.4f}  |  suma normalizada = {masc_norm.sum():.4f}')
            pesos_efectivos = masc_norm[::-1, ::-1]
        else:
            pesos_efectivos = pesos_completos
            lbl_pesos_info.config(text=f'Pesos normalizados ({tipo_mascara.get()} {tam}x{tam}): suma = {pesos_efectivos.sum():.4f}')
        # 3. Multiplicación de vecindad por kernel invertido y redondeo
        productos = matriz_i * pesos_efectivos
        suma = float(productos.sum())
        salida_clip = int(np.clip(np.rint(suma), 0, 255))
        for i in range(tam):
            for j in range(tam):
                etiquetas_prod[i][j].config(text=f'{productos[i, j]:.2f}')
        lbl_sum.config(text=f'Suma ponderada (S) = {suma:.4f}  ->  Píxel final = {salida_clip}', fg=VERDE)
        lbl_exp.config(text='Procedimiento: cada intensidad I se multiplica por su peso normalizado e invertido; la suma redondeada da el píxel final.')
    def aplicar_mascara_desde_calculo():
        if not es_pers:
            messagebox.showinfo('Aviso', 'La máscara activa es predefinida (Promedio o Gaussiana). Solo se pueden modificar los pesos de la máscara Personalizada.')
            return
        arr_w, err = leer_matriz(entradas_pes, es_pixel=False)
        if err:
            messagebox.showerror('Pesos inválidos', err.strip('[]'))
            return
        if not np.any(arr_w > 0):
            messagebox.showerror('Pesos inválidos', 'La suma de los pesos debe ser mayor a cero (al menos un peso positivo).')
            return
        if not messagebox.askyesno('Confirmar aplicación', '¿Desea aplicar estos pesos como máscara personalizada global para toda la imagen?'):
            return
        global _aplicando_desde_calculo
        try:
            _aplicando_desde_calculo = True
            establecer_matriz_codigo(arr_w.tolist())
            procesar()
            recalcular()
            messagebox.showinfo('Máscara aplicada', 'Los pesos modificados se normalizaron y se aplicaron a la imagen completa.')
        finally:
            _aplicando_desde_calculo = False
    def actualizar_desde_externo(nueva_f, nueva_c):
        f_var.set(nueva_f)
        c_var.set(nueva_c)
        cargar_vecindad()
    def actualizar_indicador_salto():
        f, c = f_var.get(), c_var.get()
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        muestreado = es_posicion_muestreada(f, c, sy, sx)
        if muestreado:
            aviso = f'[OK] La posición ({f}, {c}) SÍ se muestrea con salto {sy}x{sx}.'
        else:
            aviso = f'[!] La posición ({f}, {c}) NO se muestrea con salto {sy}x{sx}. Cálculo demostrativo.'
        lbl_salto_rel.config(text=aviso, fg=VERDE if muestreado else '#b45309',
                             bg='#f0fdf4' if muestreado else '#fefce8')
    callback_actualizar_calculo = actualizar_desde_externo
    callback_actualizar_salto = actualizar_indicador_salto
    cargar_vecindad()

def construir_interfaz():
    """Construye controles principales, panel lateral y cuadrícula de 3 vistas."""
    global raiz, nombre_imagen, estado, tipo_mascara, tamano_mascara
    global salto_horizontal, salto_vertical, paso_x_etiqueta, paso_y_etiqueta
    global tipo_ruido, intensidad_ruido, etiqueta_intensidad_ruido
    global lbl_pixel_info, marco_celdas, marco_herramientas_pers, explicacion_mascara
    global lbl_res_orig, lbl_res_red, lbl_pix_tot, lbl_pct_red, lbl_cfg_masc, lbl_cfg_saltos, lbl_t_proc
    raiz = tk.Tk()
    raiz.title('Suavizado y Reducción de Imágenes')
    raiz.geometry('1440x920')
    raiz.minsize(1040, 700)
    raiz.configure(bg=FONDO)
    # Cabecera principal moderna
    cab = tk.Frame(raiz, bg='#14233b', height=54)
    cab.pack(fill='x')
    cab.pack_propagate(False)
    tit = tk.Frame(cab, bg='#14233b')
    tit.pack(side='left', padx=16, pady=6)
    tk.Label(tit, text='Suavizado y Reducción de Imágenes', font=('Segoe UI', 15, 'bold'), bg='#14233b', fg=BLANCO).pack(anchor='w')
    btn_f = tk.Frame(cab, bg='#14233b')
    btn_f.pack(side='right', padx=14)
    for txt, cmd, col in (('Cargar imagen', cargar_imagen, AZUL), ('Procesar', procesar, AZUL),
                          ('Restaurar', restaurar, '#263a55'), ('Guardar PNG', guardar_png, VERDE)):
        tk.Button(btn_f, text=txt, command=cmd, font=('Segoe UI', 9, 'bold'), bg=col, fg=BLANCO,
                  bd=0, padx=12, pady=5, cursor='hand2').pack(side='left', padx=3)
    cuerpo = tk.Frame(raiz, bg=FONDO)
    cuerpo.pack(fill='both', expand=True, padx=8, pady=6)
    # Barra lateral de configuración y resultados
    sb = tk.Frame(cuerpo, bg=BLANCO, width=330, padx=10, pady=8, relief='solid', bd=1)
    sb.pack(side='left', fill='y', padx=(0, 8))
    sb.pack_propagate(False)
    nombre_imagen = tk.StringVar(value='Ninguna imagen')
    tk.Label(sb, textvariable=nombre_imagen, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 9)).pack(anchor='w')
    lbl_pixel_info = tk.Label(sb, text='Píxel: haga clic en la foto original', font=('Segoe UI', 8, 'italic'),
                              fg=AZUL, bg=BLANCO)
    lbl_pixel_info.pack(anchor='w', pady=(0, 6))
    # 1. Selector y cuadrícula de máscara
    tk.Label(sb, text='Filtro de suavizado', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w')
    tipo_mascara, tamano_mascara = tk.StringVar(value='Gaussiana'), tk.StringVar(value='3')
    cb_t = ttk.Combobox(sb, textvariable=tipo_mascara, values=('Gaussiana', 'Promedio', 'Moda', 'Personalizada'), state='readonly')
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
    explicacion_mascara = tk.StringVar()
    tk.Label(sb, textvariable=explicacion_mascara, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 8),
             wraplength=290).pack(anchor='w', pady=1)
    # 2. Ruido artificial de entrada, equivalente a imnoise de MATLAB.
    tk.Label(sb, text='Ruido artificial (imnoise)', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w', pady=(4, 0))
    tipo_ruido = tk.StringVar(value='Sin ruido')
    intensidad_ruido = tk.DoubleVar(value=0.0)
    etiqueta_intensidad_ruido = tk.StringVar(value='0.00')
    cb_ruido = ttk.Combobox(sb, textvariable=tipo_ruido,
                            values=('Sin ruido', 'Gaussiano', 'Sal y pimienta'), state='readonly')
    cb_ruido.pack(fill='x', pady=2)
    fila_ruido = tk.Frame(sb, bg=BLANCO)
    fila_ruido.pack(fill='x')
    tk.Label(fila_ruido, text='Intensidad/varianza:', bg=BLANCO, font=('Segoe UI', 8)).pack(side='left')
    tk.Label(fila_ruido, textvariable=etiqueta_intensidad_ruido, bg=BLANCO,
             font=('Segoe UI', 9, 'bold'), fg=AZUL).pack(side='right')
    tk.Scale(sb, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', showvalue=False,
             variable=intensidad_ruido, command=aplicar_ruido_activo,
             bg=BLANCO, bd=0, highlightthickness=0).pack(fill='x')
    # 3. Deslizadores de saltos (strides)
    tk.Label(sb, text='Saltos de reducción', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w', pady=(4, 0))
    salto_horizontal, salto_vertical = tk.IntVar(value=2), tk.IntVar(value=2)
    paso_x_etiqueta, paso_y_etiqueta = tk.StringVar(value='2'), tk.StringVar(value='2')
    for nom, var, lbl in (('Salto horizontal:', salto_horizontal, paso_x_etiqueta),
                          ('Salto vertical:', salto_vertical, paso_y_etiqueta)):
        f = tk.Frame(sb, bg=BLANCO)
        f.pack(fill='x', pady=(1, 0))
        tk.Label(f, text=nom, bg=BLANCO, font=('Segoe UI', 8)).pack(side='left')
        tk.Label(f, textvariable=lbl, bg=BLANCO, font=('Segoe UI', 9, 'bold'), fg=AZUL).pack(side='right')
        tk.Scale(sb, from_=1, to=8, orient='horizontal', showvalue=False, variable=var,
                 command=actualizar_reduccion, bg=BLANCO, bd=0, highlightthickness=0).pack(fill='x')
    # 3. Resumen de resultados
    card_res = tk.LabelFrame(sb, text='Resultados del procesamiento', font=('Segoe UI', 9, 'bold'),
                             bg='#f8fafc', fg='#1e293b', padx=8, pady=4, relief='solid', bd=1)
    card_res.pack(fill='x', pady=(4, 0))
    lbls = [tk.Label(card_res, text='-', font=('Segoe UI', 8, 'bold' if i == 3 else 'normal'),
                     bg='#f8fafc', fg=VERDE if i == 3 else (AZUL if i == 6 else '#1e293b'), anchor='w')
            for i in range(7)]
    for lb in lbls:
        lb.pack(fill='x')
    lbl_res_orig, lbl_res_red, lbl_pix_tot, lbl_pct_red, lbl_cfg_masc, lbl_cfg_saltos, lbl_t_proc = lbls
    tk.Button(sb, text='Ver cálculo paso a paso', command=mostrar_calculo,
              font=('Segoe UI', 9, 'bold'), bg='#e6edff', fg=AZUL, relief='flat', padx=6, pady=5).pack(fill='x', pady=(6, 0))
    # Mosaico de 3 vistas (Original grande izquierda, Reducción directa y con suavizado derecha)
    mos = tk.Frame(cuerpo, bg=FONDO)
    mos.pack(side='left', fill='both', expand=True)
    mos.grid_columnconfigure(0, weight=3)
    mos.grid_columnconfigure(1, weight=2)
    mos.grid_rowconfigure(0, weight=1)
    mos.grid_rowconfigure(1, weight=1)
    vistas = [
        ('original', 'IMAGEN DE ENTRADA (CON RUIDO)', 0, 0, 2, 1),
        ('directa', 'REDUCCIÓN DIRECTA (SIN SUAVIZADO)', 0, 1, 1, 1),
        ('reducida', 'REDUCCIÓN CON SUAVIZADO', 1, 1, 1, 1)
    ]
    for k, tit_txt, r, c, rspan, cspan in vistas:
        m = tk.Frame(mos, bg=BLANCO, relief='solid', bd=1, padx=3, pady=3)
        m.grid(row=r, column=c, rowspan=rspan, columnspan=cspan, sticky='nsew', padx=3, pady=3)
        m.grid_rowconfigure(1, weight=1)
        m.grid_columnconfigure(0, weight=1)
        tk.Label(m, text=tit_txt, font=('Segoe UI', 9, 'bold'), bg=BLANCO,
                 fg=AZUL if k == 'reducida' else TEXTO).grid(row=0, column=0, pady=1)
        lz = tk.Canvas(m, bg=FONDO_IMG, highlightthickness=0, cursor='crosshair' if k == 'original' else '')
        lz.grid(row=1, column=0, sticky='nsew')
        inf = tk.Label(m, text='-', font=('Segoe UI', 8), bg=BLANCO, fg=SECUNDARIO)
        inf.grid(row=2, column=0, pady=1)
        paneles[k] = (lz, inf)
        lz.bind('<Configure>', lambda e, clave_p=k: actualizar_panel(clave_p))
        if k == 'original':
            lz.bind('<Button-1>', manejar_clic_original)
    # Barra inferior de estado
    pie = tk.Frame(raiz, bg=FONDO, padx=10, pady=2)
    pie.pack(fill='x')
    estado = tk.StringVar(value='Listo.')
    tk.Label(pie, textvariable=estado, bg=FONDO, fg=SECUNDARIO, font=('Segoe UI', 8)).pack(anchor='w')
    cb_t.bind('<<ComboboxSelected>>', actualizar_mascara)
    cb_s.bind('<<ComboboxSelected>>', actualizar_mascara)
    cb_ruido.bind('<<ComboboxSelected>>', aplicar_ruido_activo)
    actualizar_mascara()
    return raiz

if __name__ == '__main__':
    construir_interfaz()
    for p in ('imagenes/paisaje.png', 'imagenes/paisaje.jpg'):
        if Path(p).exists():
            cargar_imagen(p)
            break
    raiz.mainloop()
