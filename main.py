"""Aplicacion didactica para suavizado y reduccion de imagenes RGB con mascaras y saltos.
Ejecutar: python main.py
"""

import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from procesamiento import (crear_mascara, aplicar_convolucion,
                            reducir_sin_suavizado, calcular_datos, explicar_pixel,
                            exportar_metricas_csv)

# Paleta de colores
FONDO, BLANCO, TEXTO, SECUNDARIO = '#eef2f7', '#ffffff', '#18243a', '#64748b'
AZUL, VERDE, FONDO_IMG = '#2458dc', '#059669', '#141c29'

# Matriz personalizada por defecto (editable directamente en codigo para la evaluacion)
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
nombre_imagen = tipo_mascara = tamano_mascara = None
salto_horizontal = salto_vertical = paso_x_etiqueta = paso_y_etiqueta = None
estado = explicacion_mascara = lbl_info_salto = lbl_integracion_texto = None
marco_celdas = marco_herramientas_pers = None
paneles, fotos_tk, celdas_matriz = {}, {}, []

def validar_decimal(texto):
    """Permite exclusivamente numeros: digitos y como maximo un punto o coma decimal."""
    if texto == '' or texto in ('.', ','):
        return True
    if any(c not in '0123456789.,' for c in texto):
        return False
    return texto.count('.') + texto.count(',') <= 1


def validar_pixel(texto):
    """Permite exclusivamente digitos enteros de 0 a 255 para pixeles RGB."""
    return texto == '' or (texto.isdigit() and 0 <= int(texto) <= 255)


def validar_entero_coord(texto):
    """Permite exclusivamente digitos enteros no negativos para coordenadas."""
    return texto == '' or texto.isdigit()


def establecer_matriz_codigo(nueva_matriz):
    """Permite definir o cambiar la matriz personalizada directamente desde codigo."""
    global matriz_personalizada_guardada
    matriz_personalizada_guardada = [list(f) for f in nueva_matriz]
    if tipo_mascara and tipo_mascara.get() == 'Personalizada':
        actualizar_mascara()


def cargar_imagen(ruta=None):
    """Carga imagen RGB sin alterar su resolucion real y procesa manteniendo configuracion."""
    global imagen_original, ruta_actual
    if not ruta:
        ruta = filedialog.askopenfilename(title='Seleccionar imagen', filetypes=[('Imagenes', '*.png *.jpg *.jpeg *.bmp')])
    if not ruta:
        return
    try:
        with Image.open(ruta) as f:
            imagen_original = np.array(f.convert('RGB'), dtype=np.uint8)
        ruta_actual = str(ruta)
        if nombre_imagen:
            nombre_imagen.set(Path(ruta).name)
        procesar()
    except Exception as e:
        messagebox.showerror('Error al cargar', str(e))


def guardar_png():
    """Exporta la imagen reducida real en formato PNG."""
    if imagen_reducida is None:
        messagebox.showwarning('Sin resultado', 'Primero cargue y procese una imagen.')
        return
    ruta = filedialog.asksaveasfilename(title='Guardar reducida', defaultextension='.png', filetypes=[('PNG', '*.png')])
    if ruta:
        try:
            Image.fromarray(imagen_reducida).save(ruta)
            estado.set(f'Guardada: {Path(ruta).name}')
        except Exception as e:
            messagebox.showerror('Error al guardar', str(e))


def restaurar():
    """Restaura la fotografia original sin filtros ni saltos."""
    global imagen_suavizada, imagen_reducida, imagen_directa
    if imagen_original is None:
        return
    imagen_suavizada = imagen_original.copy()
    imagen_reducida = imagen_original.copy()
    imagen_directa = imagen_original.copy()
    if estado:
        estado.set('Imagen original restaurada.')
    for c in paneles:
        actualizar_panel(c)


def mascara_desde_interfaz():
    """Obtiene mascara normalizada desde opciones o casillas editables conservando valores."""
    global matriz_personalizada_guardada
    tipo, tam = tipo_mascara.get(), int(tamano_mascara.get())
    if tipo != 'Personalizada':
        return crear_mascara(tipo, tam)
    valores = []
    for f in range(tam):
        fila = []
        for c in range(tam):
            try:
                num = float(celdas_matriz[f][c].get().strip()) if celdas_matriz[f][c].get().strip() else 0.0
            except (ValueError, IndexError):
                num = 1.0
            fila.append(num)
        valores.append(fila)
    matriz_personalizada_guardada = [list(f) for f in valores]
    matriz = np.asarray(valores, dtype=np.float64)
    if np.all(matriz <= 0):
        matriz = np.ones((tam, tam), dtype=np.float64)
    return crear_mascara(tipo, tam, matriz)


def aplicar_preset(preset):
    """Carga pesos predefinidos en la cuadricula personalizada."""
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
    """Reconstruye casillas de la mascara cargando valores editados o por defecto."""
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
                e = tk.Entry(marco_celdas, width=w_col, justify='center', font=('Consolas', 10, 'bold'), relief='solid', bd=1, **kwargs)
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
                e = tk.Entry(marco_celdas, width=w_col + 2, justify='center', font=('Consolas', 9), relief='solid', bd=1, bg='#f1f5f9', fg=SECUNDARIO)
                e.grid(row=f, column=c, padx=2, pady=2)
                e.insert(0, f'{m[f, c]:.4f}')
                e.configure(state='readonly')
                fila_w.append(e)
            celdas_matriz.append(fila_w)
        explicacion_mascara.set(f'Pesos normalizados ({tipo}): suma = {m.sum():.4f}')
    procesar()


def actualizar_reduccion(*args):
    """Calcula y visualiza la reduccion en tiempo real al mover los saltos."""
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

    nw, nh = imagen_reducida.shape[1], imagen_reducida.shape[0]
    pct = datos_actuales['reduccion']
    if lbl_info_salto:
        lbl_info_salto.config(text=f'Salida: {nw}x{nh} px (-{pct:.1f}% pixeles)')
    if estado:
        estado.set(f'[OK] Salto {sx}x{sy} -> {nw}x{nh} px | Reduccion: {pct:.1f}% pixeles ({t_ms:.1f} ms)')


def procesar():
    """Unico punto de procesamiento: normaliza pesos, convoluciona y exporta CSV."""
    global mascara_actual, imagen_suavizada
    if imagen_original is None:
        if estado:
            estado.set('Cargue una fotografia para comenzar.')
        return
    try:
        mascara_actual = mascara_desde_interfaz()
        imagen_suavizada = aplicar_convolucion(imagen_original, mascara_actual, 1, 1)
        actualizar_panel('original')
        actualizar_panel('suavizada')
        actualizar_reduccion()
    except Exception as e:
        if estado:
            estado.set(f'Error: {e}')
        return

    if tipo_mascara.get() == 'Personalizada' and lbl_integracion_texto:
        filas = ' | '.join(' '.join(f'{x:.3f}' for x in f) for f in mascara_actual)
        lbl_integracion_texto.config(text=f'Integrada: [{filas}] (suma={mascara_actual.sum():.2f})')
    try:
        sx, sy = int(salto_horizontal.get()), int(salto_vertical.get())
        exportar_metricas_csv('resultados_procesamiento.csv', Path(ruta_actual).name, datos_actuales, tipo_mascara.get(), int(tamano_mascara.get()), sx, sy)
    except Exception:
        pass


def actualizar_panel(clave):
    """Muestra la imagen en el panel reflejando la escala reducida en tiempo real."""
    img = {'original': imagen_original, 'suavizada': imagen_suavizada,
           'directa': imagen_directa, 'reducida': imagen_reducida}.get(clave)
    if img is None or imagen_original is None or clave not in paneles:
        return
    lienzo, info = paneles[clave]
    lw, lh = max(lienzo.winfo_width() - 8, 80), max(lienzo.winfo_height() - 8, 80)
    orig_h, orig_w = imagen_original.shape[:2]
    factor = min(lw / orig_w, lh / orig_h)

    foto = Image.fromarray(img)
    vw, vh = max(1, int(foto.width * factor)), max(1, int(foto.height * factor))
    vista = foto.resize((vw, vh), Image.Resampling.LANCZOS)
    foto_tk = ImageTk.PhotoImage(vista)
    fotos_tk[clave] = foto_tk
    lienzo.image = foto_tk
    lienzo.delete('all')
    lienzo.create_image(lienzo.winfo_width() // 2, lienzo.winfo_height() // 2, image=foto_tk)

    if clave in ('directa', 'reducida'):
        pct = 100 * (1 - (foto.width * foto.height) / (orig_w * orig_h))
        info.config(text=f'{foto.width} x {foto.height} px  (-{pct:.1f}%)')
    else:
        info.config(text=f'{foto.width} x {foto.height} px (100%)')


def mostrar_calculo():
    """Muestra matrices visuales del calculo paso a paso editables e interactivas."""
    if imagen_original is None or mascara_actual is None:
        messagebox.showinfo('Aviso', 'Cargue y procese una imagen primero.')
        return
    v = tk.Toplevel(raiz)
    v.title('Calculo Paso a Paso: Matrices de Convolucion (Interactivo)')
    v.geometry('920x560')
    v.configure(bg=FONDO)

    top = tk.Frame(v, bg=BLANCO, padx=10, pady=6, relief='solid', bd=1)
    top.pack(fill='x', padx=8, pady=6)
    f_var, c_var, ch_var = tk.IntVar(value=imagen_original.shape[0] // 2), tk.IntVar(value=imagen_original.shape[1] // 2), tk.IntVar(value=0)
    vcmd_dec = (v.register(validar_decimal), '%P')
    vcmd_pix = (v.register(validar_pixel), '%P')
    vcmd_coord = (v.register(validar_entero_coord), '%P')

    tk.Label(top, text='Fila:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(side='left')
    tk.Spinbox(top, from_=0, to=imagen_original.shape[0] - 1, textvariable=f_var, width=5,
               command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord).pack(side='left', padx=3)
    tk.Label(top, text='Col:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(side='left', padx=(6, 0))
    tk.Spinbox(top, from_=0, to=imagen_original.shape[1] - 1, textvariable=c_var, width=5,
               command=lambda: cargar_vecindad(), validate='key', validatecommand=vcmd_coord).pack(side='left', padx=3)

    for i, col, nom in ((0, '#dc2626', 'Rojo'), (1, '#16a34a', 'Verde'), (2, '#2563eb', 'Azul')):
        tk.Radiobutton(top, text=nom, variable=ch_var, value=i, bg=BLANCO, fg=col, font=('Segoe UI', 9, 'bold'), command=lambda: cargar_vecindad()).pack(side='left', padx=3)

    tk.Button(top, text='Cargar Pixel', command=lambda: cargar_vecindad(), bg='#263a55', fg=BLANCO, relief='flat', padx=6).pack(side='left', padx=6)
    tk.Button(top, text='Recalcular', command=lambda: recalcular(), bg=AZUL, fg=BLANCO, relief='flat', font=('Segoe UI', 9, 'bold'), padx=8).pack(side='right', padx=2)
    tk.Button(top, text='Aplicar a Imagen', command=lambda: aplicar_mascara_desde_calculo(), bg=VERDE, fg=BLANCO, relief='flat', padx=6).pack(side='right', padx=2)

    body = tk.Frame(v, bg=BLANCO, padx=10, pady=8, relief='solid', bd=1)
    body.pack(fill='both', expand=True, padx=8, pady=4)

    tk.Label(body, text='Los valores de Pixeles (I) y Mascara (W) son editables para probar calculos manuales.',
             font=('Segoe UI', 8, 'italic'), bg=BLANCO, fg=SECUNDARIO).pack(anchor='center', pady=(0, 4))

    m_row = tk.Frame(body, bg=BLANCO)
    m_row.pack(anchor='center', pady=6)

    cuadros = []
    for tit, col in (('1. Pixeles (I) [Editable]', TEXTO), ('2. Mascara (W) [Editable]', AZUL), ('3. Productos (I x W)', VERDE)):
        g = tk.Frame(m_row, bg='#f8fafc', relief='solid', bd=1, padx=6, pady=6)
        g.pack(side='left', padx=6)
        tk.Label(g, text=tit, font=('Segoe UI', 8, 'bold'), bg='#f8fafc', fg=col).pack(pady=(0, 2))
        f_sub = tk.Frame(g, bg='#f8fafc')
        f_sub.pack()
        cuadros.append(f_sub)
        if len(cuadros) < 3:
            tk.Label(m_row, text='x' if len(cuadros) == 1 else '=', font=('Segoe UI', 16, 'bold'), bg=BLANCO, fg=SECUNDARIO).pack(side='left', padx=2)

    lbl_sum = tk.Label(body, text='SUMA PONDERADA = -', font=('Segoe UI', 10, 'bold'), bg=BLANCO, fg=VERDE)
    lbl_sum.pack(pady=4)
    lbl_exp = tk.Label(body, text='', font=('Segoe UI', 8), bg=BLANCO, justify='center')
    lbl_exp.pack()

    tam = mascara_actual.shape[0]
    entradas_pix, entradas_pes, etiquetas_prod = [], [], []

    for i in range(tam):
        f_pix, f_pes, f_prod = [], [], []
        for j in range(tam):
            ep = tk.Entry(cuadros[0], width=5, justify='center', font=('Consolas', 9), relief='solid', bd=1,
                          validate='key', validatecommand=vcmd_pix)
            ep.grid(row=i, column=j, padx=1, pady=1)
            ep.bind('<KeyRelease>', lambda ev: recalcular())
            ep.bind('<Return>', lambda ev: recalcular())
            f_pix.append(ep)

            ew = tk.Entry(cuadros[1], width=6, justify='center', font=('Consolas', 9), relief='solid', bd=1, fg=AZUL,
                          validate='key', validatecommand=vcmd_dec)
            ew.grid(row=i, column=j, padx=1, pady=1)
            ew.bind('<KeyRelease>', lambda ev: recalcular())
            ew.bind('<Return>', lambda ev: recalcular())
            f_pes.append(ew)

            lp = tk.Label(cuadros[2], text='0.0', width=6, font=('Consolas', 9), relief='solid', bd=1, bg=BLANCO, fg=VERDE)
            lp.grid(row=i, column=j, padx=1, pady=1)
            f_prod.append(lp)
        entradas_pix.append(f_pix)
        entradas_pes.append(f_pes)
        etiquetas_prod.append(f_prod)

    def cargar_vecindad():
        f, c, ch = f_var.get(), c_var.get(), ch_var.get()
        if not (0 <= f < imagen_original.shape[0] and 0 <= c < imagen_original.shape[1]):
            return
        reg, pes, _, _ = explicar_pixel(imagen_original, mascara_actual, f, c)
        for i in range(tam):
            for j in range(tam):
                entradas_pix[i][j].delete(0, 'end')
                entradas_pix[i][j].insert(0, str(int(reg[i, j, ch])))
                entradas_pes[i][j].delete(0, 'end')
                entradas_pes[i][j].insert(0, f'{pes[i, j]:.4f}')
        recalcular()

    def recalcular():
        suma = 0.0
        for i in range(tam):
            for j in range(tam):
                try:
                    val_i = float(entradas_pix[i][j].get().strip())
                except ValueError:
                    val_i = 0.0
                try:
                    val_w = float(entradas_pes[i][j].get().strip())
                except ValueError:
                    val_w = 0.0
                prod = val_i * val_w
                suma += prod
                etiquetas_prod[i][j].config(text=f'{prod:.1f}')
        salida_clip = int(np.clip(round(suma), 0, 255))
        f, c = f_var.get(), c_var.get()
        lbl_sum.config(text=f'Suma ponderada = {suma:.2f}  ->  Pixel de salida = {salida_clip}')
        lbl_exp.config(text='Calculo interactivo: cada intensidad I se multiplica por su peso W y la suma da el valor final.')

    def aplicar_mascara_desde_calculo():
        valores_w = []
        for i in range(tam):
            fila = []
            for j in range(tam):
                try:
                    val = float(entradas_pes[i][j].get().strip())
                except ValueError:
                    val = 1.0
                fila.append(val)
            valores_w.append(fila)
        establecer_matriz_codigo(valores_w)
        if tipo_mascara.get() != 'Personalizada':
            tipo_mascara.set('Personalizada')
            actualizar_mascara()
        procesar()
        messagebox.showinfo('Aplicada', 'Los pesos modificados se aplicaron a la imagen.')

    cargar_vecindad()


def construir_interfaz():
    """Construye controles principales con un unico boton de procesamiento."""
    global raiz, nombre_imagen, estado, tipo_mascara, tamano_mascara
    global salto_horizontal, salto_vertical, paso_x_etiqueta, paso_y_etiqueta, lbl_info_salto
    global marco_celdas, marco_herramientas_pers, lbl_integracion_texto, explicacion_mascara

    raiz = tk.Tk()
    raiz.title('Taller de IA | Suavizado y Reduccion de Imagenes')
    raiz.geometry('1440x900')
    raiz.minsize(1020, 680)
    raiz.configure(bg=FONDO)

    # Cabecera con UN SOLO boton de procesamiento
    cab = tk.Frame(raiz, bg='#14233b', height=65)
    cab.pack(fill='x')
    cab.pack_propagate(False)
    tit = tk.Frame(cab, bg='#14233b')
    tit.pack(side='left', padx=16, pady=6)
    tk.Label(tit, text='Suavizado y Reduccion de Imagenes', font=('Segoe UI', 15, 'bold'), bg='#14233b', fg=BLANCO).pack(anchor='w')
    tk.Label(tit, text='Convolucion manual + saltos (sin funciones automaticas)', font=('Segoe UI', 9), bg='#14233b', fg='#b6c7df').pack(anchor='w')

    btn_f = tk.Frame(cab, bg='#14233b')
    btn_f.pack(side='right', padx=14)
    for txt, cmd, col in (('Cargar imagen', cargar_imagen, AZUL), ('Procesar', procesar, AZUL),
                          ('Restaurar', restaurar, '#263a55'), ('Guardar PNG', guardar_png, VERDE)):
        tk.Button(btn_f, text=txt, command=cmd, font=('Segoe UI', 9, 'bold'), bg=col, fg=BLANCO, bd=0, padx=12, pady=5, cursor='hand2').pack(side='left', padx=3)

    cuerpo = tk.Frame(raiz, bg=FONDO)
    cuerpo.pack(fill='both', expand=True, padx=8, pady=6)

    # Barra lateral
    sb = tk.Frame(cuerpo, bg=BLANCO, width=320, padx=10, pady=8, relief='solid', bd=1)
    sb.pack(side='left', fill='y', padx=(0, 8))
    sb.pack_propagate(False)

    nombre_imagen = tk.StringVar(value='Ninguna imagen')
    tk.Label(sb, text='Fotografia:', font=('Segoe UI', 9, 'bold'), bg=BLANCO).pack(anchor='w')
    tk.Label(sb, textvariable=nombre_imagen, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 6))

    tk.Label(sb, text='1. Mascara de suavizado', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w')
    tipo_mascara, tamano_mascara = tk.StringVar(value='Gaussiana'), tk.StringVar(value='3')
    cb_t = ttk.Combobox(sb, textvariable=tipo_mascara, values=('Gaussiana', 'Promedio', 'Personalizada'), state='readonly')
    cb_t.pack(fill='x', pady=2)
    cb_s = ttk.Combobox(sb, textvariable=tamano_mascara, values=('3', '5', '7'), state='readonly')
    cb_s.pack(fill='x', pady=2)

    c_box = tk.Frame(sb, bg='#f8fafc', relief='solid', bd=1, padx=4, pady=4)
    c_box.pack(fill='x', pady=4)
    marco_celdas = tk.Frame(c_box, bg='#f8fafc')
    marco_celdas.pack(anchor='center')

    # Herramientas de matriz personalizada (sin boton duplicado, solo presets e integracion)
    marco_herramientas_pers = tk.Frame(c_box, bg='#f8fafc')
    pr_f = tk.Frame(marco_herramientas_pers, bg='#f8fafc')
    pr_f.pack(fill='x', pady=(2, 4))
    tk.Button(pr_f, text='1s', command=lambda: aplicar_preset('unos'), font=('Segoe UI', 7), relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    tk.Button(pr_f, text='1-2-1', command=lambda: aplicar_preset('gauss'), font=('Segoe UI', 7), relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    tk.Button(pr_f, text='Centro 5', command=lambda: aplicar_preset('centro'), font=('Segoe UI', 7), relief='flat', bg='#e2e8f0').pack(side='left', expand=True, fill='x', padx=1)
    lbl_integracion_texto = tk.Label(marco_herramientas_pers, text='Integrada: -', font=('Segoe UI', 7), fg=AZUL, bg='#f8fafc', wraplength=280, justify='left')
    lbl_integracion_texto.pack(anchor='w', pady=1)

    explicacion_mascara = tk.StringVar()
    tk.Label(sb, textvariable=explicacion_mascara, fg=SECUNDARIO, bg=BLANCO, font=('Segoe UI', 8), wraplength=290).pack(anchor='w', pady=2)

    tk.Label(sb, text='2. Saltos de reduccion', font=('Segoe UI', 10, 'bold'), bg=BLANCO).pack(anchor='w', pady=(6, 0))
    salto_horizontal, salto_vertical = tk.IntVar(value=2), tk.IntVar(value=2)
    paso_x_etiqueta, paso_y_etiqueta = tk.StringVar(value='2'), tk.StringVar(value='2')

    for nom, var, lbl in (('Salto horizontal:', salto_horizontal, paso_x_etiqueta), ('Salto vertical:', salto_vertical, paso_y_etiqueta)):
        f = tk.Frame(sb, bg=BLANCO)
        f.pack(fill='x', pady=(2, 0))
        tk.Label(f, text=nom, bg=BLANCO, font=('Segoe UI', 8)).pack(side='left')
        tk.Label(f, textvariable=lbl, bg=BLANCO, font=('Segoe UI', 9, 'bold'), fg=AZUL).pack(side='right')
        sc = tk.Scale(sb, from_=1, to=8, orient='horizontal', showvalue=False, variable=var,
                      command=actualizar_reduccion, bg=BLANCO, bd=0, highlightthickness=0)
        sc.pack(fill='x')

    lbl_info_salto = tk.Label(sb, text='Salida: -', font=('Segoe UI', 8), fg=SECUNDARIO, bg=BLANCO)
    lbl_info_salto.pack(anchor='w', pady=2)
    tk.Button(sb, text='Ver calculo paso a paso', command=mostrar_calculo, font=('Segoe UI', 9, 'bold'), bg='#e6edff', fg=AZUL, relief='flat', padx=6, pady=5).pack(fill='x', pady=(10, 0))

    # Mosaico de 4 vistas
    mos = tk.Frame(cuerpo, bg=FONDO)
    mos.pack(side='left', fill='both', expand=True)
    mos.grid_columnconfigure((0, 1), weight=1, uniform='img')
    mos.grid_rowconfigure((0, 1), weight=1)

    vistas = [('original', 'IMAGEN ORIGINAL'), ('suavizada', 'IMAGEN SUAVIZADA'),
              ('directa', 'REDUCCION DIRECTA'), ('reducida', 'REDUCCION CON SUAVIZADO')]
    for idx, (k, tit_txt) in enumerate(vistas):
        m = tk.Frame(mos, bg=BLANCO, relief='solid', bd=1, padx=3, pady=3)
        m.grid(row=idx // 2, column=idx % 2, sticky='nsew', padx=3, pady=3)
        m.grid_rowconfigure(1, weight=1)
        m.grid_columnconfigure(0, weight=1)
        tk.Label(m, text=tit_txt, font=('Segoe UI', 9, 'bold'), bg=BLANCO, fg=AZUL if k == 'reducida' else TEXTO).grid(row=0, column=0, pady=1)
        lz = tk.Canvas(m, bg=FONDO_IMG, highlightthickness=0)
        lz.grid(row=1, column=0, sticky='nsew')
        inf = tk.Label(m, text='-', font=('Segoe UI', 8), bg=BLANCO, fg=SECUNDARIO)
        inf.grid(row=2, column=0, pady=1)
        paneles[k] = (lz, inf)
        lz.bind('<Configure>', lambda e, c=k: actualizar_panel(c))

    # Pie
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
