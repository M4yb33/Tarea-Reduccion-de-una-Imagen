"""Aplicación didáctica para suavizar y reducir imágenes RGB con máscaras y saltos.

Ejecutar: python main.py
El algoritmo está en procesamiento.py; este archivo muestra controles e imágenes.
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

# Colores de la interfaz (no forman parte del procesamiento de las fotografías).
FONDO = '#eef2f7'
BLANCO = '#ffffff'
TEXTO = '#18243a'
SECUNDARIO = '#64748b'
AZUL = '#2458dc'
FONDO_IMAGEN = '#141c29'

# Estado sencillo, compartido entre los controles y los cuatro paneles.
raiz = None
imagen_original = None
imagen_suavizada = None
imagen_reducida = None
imagen_directa = None
mascara_actual = None
datos_actuales = None
ruta_actual = ''
paneles = {}
fotos_tk = {}
celdas_matriz = []  # Lista bidimensional de Entry widgets para la cuadrícula
actualizacion_pendiente = None
cargando_controles = False


def cargar_imagen(ruta=None):
    """Lee una fotografía RGB sin cambiar su tamaño real."""
    global imagen_original, ruta_actual
    if not ruta:
        ruta = filedialog.askopenfilename(
            title='Seleccionar imagen',
            filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp')]
        )
    if not ruta:
        return
    try:
        with Image.open(ruta) as archivo:
            nueva = np.array(archivo.convert('RGB'), dtype=np.uint8)
    except Exception as error:
        messagebox.showerror('No se pudo cargar', str(error))
        return
    imagen_original = nueva
    ruta_actual = str(ruta)
    nombre_imagen.set(Path(ruta).name)
    procesar()


def mascara_desde_interfaz():
    """Obtiene una máscara normalizada leyendo la cuadrícula de celdas."""
    tipo = tipo_mascara.get()
    tamano = int(tamano_mascara.get())
    if tipo != 'Personalizada':
        return crear_mascara(tipo, tamano)

    # Leemos cada celda (cuadro) de la matriz interactiva
    valores = []
    for fila in range(tamano):
        fila_valores = []
        for col in range(tamano):
            if fila < len(celdas_matriz) and col < len(celdas_matriz[fila]):
                texto = celdas_matriz[fila][col].get().strip()
                try:
                    numero = float(texto) if texto else 0.0
                except ValueError:
                    numero = 0.0
            else:
                numero = 1.0
            fila_valores.append(numero)
        valores.append(fila_valores)

    return crear_mascara(tipo, tamano, valores)


def rellenar_mascara_con_unos():
    """Rellena todas las casillas de la máscara personalizada con el valor 1."""
    tamano = int(tamano_mascara.get())
    for fila in range(tamano):
        for col in range(tamano):
            if fila < len(celdas_matriz) and col < len(celdas_matriz[fila]):
                celdas_matriz[fila][col].delete(0, 'end')
                celdas_matriz[fila][col].insert(0, '1')
    solicitar_procesamiento()


def actualizar_mascara(*args):
    """Reconstruye la cuadrícula de cuadros para cada número de la matriz."""
    global celdas_matriz
    if cargando_controles or raiz is None:
        return

    tipo = tipo_mascara.get()
    tamano = int(tamano_mascara.get())

    # Limpiamos los cuadros anteriores de la cuadrícula
    for widget in marco_cuadricula.winfo_children():
        widget.destroy()
    celdas_matriz = []

    # Configuración de ancho según el tamaño de la máscara
    ancho_celda = 5 if tamano == 3 else (4 if tamano == 5 else 3)

    if tipo == 'Personalizada':
        # Casillas editables individuales para cada peso
        for fila in range(tamano):
            fila_widgets = []
            for col in range(tamano):
                cuadro = tk.Entry(
                    marco_cuadricula,
                    width=ancho_celda,
                    justify='center',
                    font=('Consolas', 10),
                    relief='solid',
                    bd=1,
                    bg=BLANCO,
                    fg=TEXTO
                )
                cuadro.grid(row=fila, column=col, padx=2, pady=2)
                cuadro.insert(0, '1')
                cuadro.bind('<KeyRelease>', solicitar_procesamiento)
                fila_widgets.append(cuadro)
            celdas_matriz.append(fila_widgets)

        boton_unos.pack(anchor='center', pady=(4, 2))
        explicacion_mascara.set('Editable: ingrese los pesos en cada cuadro (se normalizan automáticamente).')
    else:
        # Modo lectura para Promedio y Gaussiana
        boton_unos.pack_forget()
        mascara = crear_mascara(tipo, tamano)
        for fila in range(tamano):
            fila_widgets = []
            for col in range(tamano):
                cuadro = tk.Entry(
                    marco_cuadricula,
                    width=ancho_celda,
                    justify='center',
                    font=('Consolas', 9),
                    relief='solid',
                    bd=1,
                    bg='#f1f5f9',
                    fg=SECUNDARIO
                )
                cuadro.grid(row=fila, column=col, padx=2, pady=2)
                cuadro.insert(0, f'{mascara[fila, col]:.3f}')
                cuadro.configure(state='readonly')
                fila_widgets.append(cuadro)
            celdas_matriz.append(fila_widgets)

        explicacion_mascara.set(f'Pesos normalizados en cada celda: suma = {mascara.sum():.4f}')

    solicitar_procesamiento()


def solicitar_procesamiento(*args):
    """Evita procesar muchas veces durante un mismo arrastre del slider."""
    global actualizacion_pendiente
    if cargando_controles or raiz is None:
        return
    paso_x_etiqueta.set(str(int(salto_horizontal.get())))
    paso_y_etiqueta.set(str(int(salto_vertical.get())))
    if actualizacion_pendiente is not None:
        raiz.after_cancel(actualizacion_pendiente)
    actualizacion_pendiente = raiz.after(180, procesar)


def procesar():
    """Aplica la máscara con padding de ceros interno; reduce según los saltos."""
    global imagen_suavizada, imagen_reducida, imagen_directa, mascara_actual, datos_actuales
    global actualizacion_pendiente
    if actualizacion_pendiente is not None:
        raiz.after_cancel(actualizacion_pendiente)
        actualizacion_pendiente = None
    if imagen_original is None:
        estado.set('Cargue una fotografía para comenzar.')
        return
    try:
        mascara = mascara_desde_interfaz()
        salto_x = int(salto_horizontal.get())
        salto_y = int(salto_vertical.get())

        inicio = time.perf_counter()
        # Padding de ceros aplicado internamente en aplicar_convolucion
        suavizada = aplicar_convolucion(imagen_original, mascara, 1, 1)
        reducida = aplicar_convolucion(imagen_original, mascara, salto_y, salto_x)
        directa = reducir_sin_suavizado(imagen_original, salto_y, salto_x)
        transcurrido = (time.perf_counter() - inicio) * 1000
        datos = calcular_datos(imagen_original, suavizada, reducida, transcurrido)
    except Exception as error:
        estado.set('Revise la máscara: ' + str(error))
        etiqueta_estado.configure(fg='#b91c1c')
        return

    mascara_actual = mascara
    imagen_suavizada = suavizada
    imagen_reducida = reducida
    imagen_directa = directa
    datos_actuales = datos
    etiqueta_estado.configure(fg=SECUNDARIO)

    # Generación automática del CSV con las métricas comprobables
    try:
        exportar_metricas_csv(
            'resultados_procesamiento.csv',
            Path(ruta_actual).name,
            datos,
            tipo_mascara.get(),
            int(tamano_mascara.get()),
            salto_x,
            salto_y
        )
        estado.set(f'Procesado en {datos["tiempo_ms"]:.1f} ms. Métricas guardadas en "resultados_procesamiento.csv".')
    except Exception:
        estado.set(f'Procesado en {datos["tiempo_ms"]:.1f} ms. Para ver un producto, pulse «Ver cálculo».')

    for clave in paneles:
        actualizar_panel(clave)


def restaurar():
    """Recupera la fotografía original, sin suavizado ni reducción."""
    global imagen_suavizada, imagen_reducida, imagen_directa
    global actualizacion_pendiente
    if imagen_original is None:
        return
    if actualizacion_pendiente is not None:
        raiz.after_cancel(actualizacion_pendiente)
        actualizacion_pendiente = None
    imagen_suavizada = imagen_original.copy()
    imagen_reducida = imagen_original.copy()
    imagen_directa = imagen_original.copy()
    estado.set('Imagen original restaurada. Pulse «Procesar» para aplicar los ajustes visibles.')
    for clave in paneles:
        actualizar_panel(clave)


def guardar_png():
    """Exporta la matriz REAL, no la fotografía ampliada del panel."""
    if imagen_reducida is None:
        messagebox.showwarning('Sin resultado', 'Primero cargue y procese una imagen.')
        return
    ruta = filedialog.asksaveasfilename(
        title='Guardar imagen reducida', defaultextension='.png',
        filetypes=[('Imagen PNG', '*.png')]
    )
    if ruta:
        try:
            Image.fromarray(imagen_reducida).save(ruta)
            estado.set(f'Imagen reducida guardada: {Path(ruta).name}')
        except Exception as error:
            messagebox.showerror('No se pudo guardar', str(error))


def guardar_csv():
    """Permite exportar explícitamente el archivo CSV con las métricas calculadas."""
    if datos_actuales is None:
        messagebox.showwarning('Sin métricas', 'Primero cargue y procese una imagen para generar métricas.')
        return
    ruta = filedialog.asksaveasfilename(
        title='Guardar resultados en CSV',
        defaultextension='.csv',
        filetypes=[('Archivo CSV', '*.csv')]
    )
    if ruta:
        try:
            exportar_metricas_csv(
                ruta,
                Path(ruta_actual).name,
                datos_actuales,
                tipo_mascara.get(),
                int(tamano_mascara.get()),
                int(salto_horizontal.get()),
                int(salto_vertical.get())
            )
            estado.set(f'Métricas exportadas correctamente en: {Path(ruta).name}')
            messagebox.showinfo('CSV Guardado', f'Métricas exportadas exitosamente en:\n{ruta}')
        except Exception as error:
            messagebox.showerror('Error al exportar CSV', str(error))


def actualizar_panel(clave):
    """Adapta la VISUALIZACIÓN a cada recuadro, sin modificar la matriz real."""
    imagen = {
        'original': imagen_original,
        'suavizada': imagen_suavizada,
        'directa': imagen_directa,
        'reducida': imagen_reducida,
    }[clave]
    if imagen is None:
        return
    lienzo, informacion = paneles[clave]
    ancho = max(lienzo.winfo_width() - 20, 100)
    alto = max(lienzo.winfo_height() - 20, 90)
    foto = Image.fromarray(imagen)
    factor = min(ancho / foto.width, alto / foto.height)
    ancho_vista = max(1, int(foto.width * factor))
    alto_vista = max(1, int(foto.height * factor))
    # EXCLUSIVAMENTE para mostrar más grande o más pequeño dentro de Tkinter.
    # La imagen reducida guardada NO pasa por esta operación.
    vista = foto.resize((ancho_vista, alto_vista), Image.Resampling.LANCZOS)
    fotos_tk[clave] = ImageTk.PhotoImage(vista)
    lienzo.delete('all')
    lienzo.create_image(lienzo.winfo_width() // 2, lienzo.winfo_height() // 2,
                        image=fotos_tk[clave])
    informacion.configure(text=f'{foto.width} × {foto.height} píxeles reales')


def mostrar_calculo():
    """Desglosa la convolución de un píxel real, útil durante la defensa."""
    if imagen_original is None or mascara_actual is None:
        return
    ventana = tk.Toplevel(raiz)
    ventana.title('Cómo se calcula un píxel — multiplicación y suma')
    ventana.geometry('740x660')
    ventana.configure(bg=FONDO)
    ventana.transient(raiz)
    controles = tk.Frame(ventana, bg=FONDO)
    controles.pack(fill='x', padx=18, pady=12)
    tk.Label(controles, text='Fila original:', bg=FONDO).pack(side='left')
    fila_var = tk.IntVar(value=imagen_original.shape[0] // 2)
    columna_var = tk.IntVar(value=imagen_original.shape[1] // 2)
    tk.Spinbox(controles, from_=0, to=imagen_original.shape[0]-1,
               textvariable=fila_var, width=8).pack(side='left', padx=7)
    tk.Label(controles, text='Columna original:', bg=FONDO).pack(side='left')
    tk.Spinbox(controles, from_=0, to=imagen_original.shape[1]-1,
               textvariable=columna_var, width=8).pack(side='left', padx=7)
    caja = tk.Text(ventana, font=('Consolas', 10), wrap='none', bg=BLANCO,
                   fg=TEXTO, relief='flat', padx=14, pady=12)
    caja.pack(fill='both', expand=True, padx=18, pady=(0, 18))

    def presentar():
        fila = fila_var.get()
        columna = columna_var.get()
        if not (0 <= fila < imagen_original.shape[0] and
                0 <= columna < imagen_original.shape[1]):
            return
        # Padding de ceros aplicado internamente
        region, pesos, productos, salida = explicar_pixel(
            imagen_original, mascara_actual, fila, columna)
        texto = [f'PÍXEL DE LA IMAGEN ORIGINAL: fila {fila}, columna {columna}',
                 f'Máscara de {len(pesos)} × {len(pesos)} (pesos invertidos para convolución)',
                 f'Suma de pesos: {pesos.sum():.5f}',
                 f'Salto horizontal/vertical actual: {int(salto_horizontal.get())} / '
                 f'{int(salto_vertical.get())}',
                 'NOTA: este cálculo puede realizarse para cualquier posición original.', '']
        for indice, nombre in enumerate(('ROJO', 'VERDE', 'AZUL')):
            texto += [f'--- CANAL {nombre} ---', 'Vecindad con padding de ceros (intensidades):']
            for fila_p in region[:, :, indice]:
                texto.append(' '.join(f'{valor:7d}' for valor in fila_p))
            texto.append('Máscara normalizada:')
            for fila_p in pesos:
                texto.append(' '.join(f'{valor:7.4f}' for valor in fila_p))
            texto.append('Productos píxel × peso:')
            for fila_p in productos[:, :, indice]:
                texto.append(' '.join(f'{valor:7.2f}' for valor in fila_p))
            texto.append(f'SUMA = {productos[:, :, indice].sum():.3f}')
            texto.append(f'Valor final redondeado = {int(salida[indice])}')
            texto.append('')
        texto.append('El proceso completo repite estos productos y sumas en todas')
        texto.append('las posiciones, avanzando según los saltos definidos.')
        caja.configure(state='normal')
        caja.delete('1.0', 'end')
        caja.insert('end', '\n'.join(texto))
        caja.configure(state='disabled')

    tk.Button(controles, text='Mostrar multiplicaciones', command=presentar,
              bg=AZUL, fg=BLANCO, relief='flat', padx=10).pack(side='left', padx=8)
    presentar()


def construir_interfaz():
    """Crea todos los controles; no necesita clases ni frameworks complejos."""
    global raiz, nombre_imagen, estado, etiqueta_estado
    global tipo_mascara, tamano_mascara, salto_horizontal, salto_vertical
    global marco_cuadricula, boton_unos, explicacion_mascara
    global paso_x_etiqueta, paso_y_etiqueta

    raiz = tk.Tk()
    raiz.title('Taller de IA | Suavizado y reducción manual por convolución')
    raiz.geometry('1500x950')
    raiz.minsize(1030, 730)
    raiz.configure(bg=FONDO)

    estilo = ttk.Style()
    estilo.theme_use('clam')
    estilo.configure('Elegir.TCombobox', font=('Segoe UI', 10), padding=4)
    estilo.configure('Accion.TButton', font=('Segoe UI', 10), padding=7)

    # Encabezado
    cabecera = tk.Frame(raiz, bg='#14233b', height=83)
    cabecera.pack(fill='x')
    cabecera.pack_propagate(False)
    titulos = tk.Frame(cabecera, bg='#14233b')
    titulos.pack(side='left', padx=20, pady=12)
    tk.Label(titulos, text='Suavizado y reducción de imágenes',
             font=('Segoe UI', 20, 'bold'), bg='#14233b', fg=BLANCO).pack(anchor='w')
    tk.Label(titulos, text='Máscaras de convolución + saltos configurables · sin reducción automática',
             font=('Segoe UI', 10), bg='#14233b', fg='#b6c7df').pack(anchor='w')
    herramientas = tk.Frame(cabecera, bg='#14233b')
    herramientas.pack(side='right', padx=16)

    # Botones principales en el encabezado, incluyendo la exportación a CSV
    for texto, funcion in (('Cargar imagen', cargar_imagen),
                           ('Procesar', procesar),
                           ('Restaurar original', restaurar),
                           ('Guardar PNG', guardar_png),
                           ('Guardar CSV', guardar_csv)):
        tk.Button(herramientas, text=texto, command=funcion,
                  font=('Segoe UI', 10, 'bold'),
                  bg=AZUL if texto == 'Cargar imagen' else ('#059669' if texto == 'Guardar CSV' else '#263a55'),
                  fg=BLANCO, activebackground='#3466e3', activeforeground=BLANCO,
                  bd=0, cursor='hand2', padx=13, pady=10).pack(side='left', padx=4)

    cuerpo = tk.Frame(raiz, bg=FONDO)
    cuerpo.pack(fill='both', expand=True, padx=12, pady=12)

    # Izquierda: controles organizados
    barra = tk.Frame(cuerpo, bg=BLANCO, width=358)
    barra.pack(side='left', fill='y', padx=(0, 12))
    barra.pack_propagate(False)
    scroll_canvas = tk.Canvas(barra, bg=BLANCO, highlightthickness=0, width=343)
    scroll_canvas.pack(side='left', fill='both', expand=True)
    desplazador = ttk.Scrollbar(barra, orient='vertical', command=scroll_canvas.yview)
    desplazador.pack(side='right', fill='y')
    scroll_canvas.configure(yscrollcommand=desplazador.set)
    control = tk.Frame(scroll_canvas, bg=BLANCO, padx=15, pady=12)
    id_control = scroll_canvas.create_window((0, 0), window=control, anchor='nw')
    control.bind('<Configure>', lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all')))
    scroll_canvas.bind('<Configure>', lambda e: scroll_canvas.itemconfigure(id_control, width=e.width))

    def titulo_seccion(texto):
        tk.Label(control, text=texto, bg=BLANCO, fg=TEXTO,
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(13, 5))

    def texto_campo(texto):
        tk.Label(control, text=texto, bg=BLANCO, fg=SECUNDARIO,
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(9, 3))

    nombre_imagen = tk.StringVar(value='Ninguna imagen cargada')
    tk.Label(control, text='Fotografía seleccionada', bg=BLANCO, fg=TEXTO,
             font=('Segoe UI', 11, 'bold')).pack(anchor='w')
    tk.Label(control, textvariable=nombre_imagen, bg=BLANCO, fg=SECUNDARIO,
             wraplength=305, anchor='w', justify='left').pack(fill='x', pady=(3, 4))

    # 1. Configuración de Máscara con cuadrícula interactiva
    titulo_seccion('1. Elegir máscara de suavizado')
    tipo_mascara = tk.StringVar(value='Gaussiana')
    tamano_mascara = tk.StringVar(value='3')
    texto_campo('Tipo de máscara')
    selector_tipo = ttk.Combobox(control, textvariable=tipo_mascara,
                                 values=('Gaussiana', 'Promedio', 'Personalizada'),
                                 state='readonly', style='Elegir.TCombobox')
    selector_tipo.pack(fill='x')
    texto_campo('Tamaño (siempre impar)')
    selector_tamano = ttk.Combobox(control, textvariable=tamano_mascara,
                                   values=('3', '5', '7'), state='readonly',
                                   style='Elegir.TCombobox')
    selector_tamano.pack(fill='x')

    # Cuadrícula interactiva: cada número en su propio cuadro (Entry)
    texto_campo('Matriz de pesos (cada número en un cuadro):')
    marco_cuadricula_exterior = tk.Frame(control, bg='#f8fafc', relief='solid', bd=1, padx=6, pady=6)
    marco_cuadricula_exterior.pack(fill='x', pady=(3, 3))

    marco_cuadricula = tk.Frame(marco_cuadricula_exterior, bg='#f8fafc')
    marco_cuadricula.pack(anchor='center')

    boton_unos = tk.Button(
        marco_cuadricula_exterior,
        text='Rellenar casillas con 1s',
        command=rellenar_mascara_con_unos,
        font=('Segoe UI', 8),
        bg='#e2e8f0',
        fg=TEXTO,
        relief='flat',
        padx=8,
        pady=2,
        cursor='hand2'
    )

    explicacion_mascara = tk.StringVar()
    tk.Label(control, textvariable=explicacion_mascara, bg=BLANCO,
             fg=SECUNDARIO, font=('Segoe UI', 8), wraplength=305,
             justify='left').pack(anchor='w', pady=(3, 0))

    # 2. Configuración de Saltos (Strides)
    titulo_seccion('2. Elegir cuánto avanzar')
    tk.Label(control, text='Los saltos determinan la resolución final.',
             bg=BLANCO, fg=SECUNDARIO, wraplength=304).pack(anchor='w')
    salto_horizontal = tk.IntVar(value=2)
    salto_vertical = tk.IntVar(value=2)
    paso_x_etiqueta = tk.StringVar(value='2')
    paso_y_etiqueta = tk.StringVar(value='2')
    for nombre, variable, etiqueta in (
        ('Salto horizontal (columnas)', salto_horizontal, paso_x_etiqueta),
        ('Salto vertical (filas)', salto_vertical, paso_y_etiqueta),
    ):
        fila = tk.Frame(control, bg=BLANCO)
        fila.pack(fill='x', pady=(12, 0))
        tk.Label(fila, text=nombre, bg=BLANCO, fg=TEXTO).pack(side='left')
        tk.Label(fila, textvariable=etiqueta, bg=BLANCO, fg=AZUL,
                 font=('Segoe UI', 11, 'bold')).pack(side='right')
        tk.Scale(control, from_=1, to=8, orient='horizontal', showvalue=False,
                 variable=variable, resolution=1, command=solicitar_procesamiento,
                 bg=BLANCO, highlightthickness=0, troughcolor='#d5e1f7',
                 activebackground=AZUL, bd=0).pack(fill='x')

    # Botón didáctico para la defensa
    tk.Button(control, text='Ver cálculo de un píxel, paso a paso',
              command=mostrar_calculo, font=('Segoe UI', 10, 'bold'), bg='#e6edff',
              fg=AZUL, bd=0, cursor='hand2', padx=10, pady=10,
              wraplength=295).pack(fill='x', pady=(20, 0))

    # Derecha: cuatro vistas con dimensiones REALES debajo de cada imagen.
    mosaico = tk.Frame(cuerpo, bg=FONDO)
    mosaico.pack(side='left', fill='both', expand=True)
    mosaico.grid_columnconfigure(0, weight=1, uniform='imagenes')
    mosaico.grid_columnconfigure(1, weight=1, uniform='imagenes')
    mosaico.grid_rowconfigure(0, weight=1)
    mosaico.grid_rowconfigure(1, weight=1)
    tarjetas = [
        ('original', 'IMAGEN ORIGINAL', 'Referencia sin procesar'),
        ('suavizada', 'IMAGEN SUAVIZADA', 'Máscara aplicada, tamaño original'),
        ('directa', 'REDUCCIÓN DIRECTA', 'Solo saltos · sin suavizado'),
        ('reducida', 'REDUCCIÓN CON SUAVIZADO', 'Máscara + saltos · resultado final'),
    ]
    for indice, (clave, titulo, descripcion) in enumerate(tarjetas):
        marco = tk.Frame(mosaico, bg=BLANCO, bd=0)
        marco.grid(row=indice//2, column=indice%2, sticky='nsew',
                   padx=5, pady=5)
        marco.grid_rowconfigure(2, weight=1)
        marco.grid_columnconfigure(0, weight=1)
        tk.Label(marco, text=titulo, bg=BLANCO, fg=AZUL if clave=='reducida' else TEXTO,
                 font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, pady=(12, 2))
        tk.Label(marco, text=descripcion, bg=BLANCO, fg=SECUNDARIO,
                 font=('Segoe UI', 9)).grid(row=1, column=0, pady=(0, 7))
        lienzo = tk.Canvas(marco, bg=FONDO_IMAGEN, highlightthickness=0, height=210)
        lienzo.grid(row=2, column=0, sticky='nsew', padx=10)
        detalle = tk.Label(marco, text='—', bg=BLANCO, fg=SECUNDARIO,
                           font=('Segoe UI', 9))
        detalle.grid(row=3, column=0, pady=10)
        paneles[clave] = (lienzo, detalle)
        lienzo.bind('<Configure>', lambda evento, c=clave: actualizar_panel(c))

    # Pie de página y barra de estado
    pie = tk.Frame(raiz, bg=FONDO)
    pie.pack(fill='x', padx=21, pady=(0, 8))
    estado = tk.StringVar(value='Esperando una fotografía...')
    etiqueta_estado = tk.Label(pie, textvariable=estado, bg=FONDO, fg=SECUNDARIO,
                               font=('Segoe UI', 9))
    etiqueta_estado.pack(anchor='w')
    tk.Label(pie, text='El padding de ceros se aplica internamente. La ampliación de las vistas es solo visual; los archivos conservan los píxeles reales.',
             bg=FONDO, fg=SECUNDARIO, font=('Segoe UI', 8)).pack(anchor='w', pady=(3, 0))

    selector_tipo.bind('<<ComboboxSelected>>', actualizar_mascara)
    selector_tamano.bind('<<ComboboxSelected>>', actualizar_mascara)
    actualizar_mascara()
    return raiz


if __name__ == '__main__':
    construir_interfaz()
    ejemplo = Path(__file__).resolve().parent / 'imagenes' / 'paisaje.png'
    if not ejemplo.exists():
        ejemplo = Path(__file__).resolve().parent / 'imagenes' / 'paisaje.jpg'
    if ejemplo.exists():
        cargar_imagen(str(ejemplo))
    raiz.mainloop()
