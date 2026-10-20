"""
=============================================================================
PROYECTO: Taller de Reducción y Suavizado de Imágenes con Máscara
ASIGNATURA: Inteligencia Artificial
ARCHIVO: main.py
DESCRIPCIÓN:
    Interfaz gráfica sencilla desarrollada con Tkinter.
    Permite:
    1. Cargar una imagen (JPG, PNG).
    2. Seleccionar tipo de máscara: Promedio, Gaussiana o Personalizada.
    3. Seleccionar el tamaño de la máscara (3x3, 5x5, 7x7).
    4. Configurar saltos (strides) horizontal y vertical.
    5. Visualizar la matriz numérica de la máscara.
    6. Mostrar la imagen original y la reducida con sus dimensiones.
    7. Guardar la imagen reducida.
    8. Comparar visual y numéricamente la reducción con suavizado vs directa.

    Diseñado sin programación orientada a objetos (sin clases ni 'self')
    para máxima facilidad de comprensión y defensa por los estudiantes.
=============================================================================
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# Importamos las funciones matemáticas de nuestro archivo procesamiento.py
import procesamiento


# =============================================================================
# VARIABLES GLOBALES DE ESTADO (Fáciles de seguir y explicar)
# =============================================================================
imagen_original = None       # Matriz NumPy de la imagen original (alto, ancho, 3)
imagen_procesada = None      # Matriz NumPy de la imagen reducida con suavizado
imagen_directa = None        # Matriz NumPy de la imagen reducida directa (sin filtro)
mascara_actual = None        # Matriz NumPy de la máscara actual

# Referencias para evitar que el recolector de basura de Python borre las imágenes en pantalla
foto_original_tk = None
foto_procesada_tk = None
foto_comp_suavizada_tk = None
foto_comp_directa_tk = None


# =============================================================================
# FUNCIONES AUXILIARES PARA MOSTRAR IMÁGENES EN LA INTERFAZ
# =============================================================================

def convertir_matriz_a_foto_tk(matriz, ancho_max=380, alto_max=320):
    """
    Convierte una matriz NumPy a un objeto PhotoImage de Tkinter ajustado al visor.
    """
    if matriz is None:
        return None

    # Aseguramos que los valores estén en [0, 255] y tipo uint8
    matriz_uint8 = np.clip(matriz, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(matriz_uint8)

    # Calculamos el factor de escala para mantener la proporción
    ancho_orig, alto_orig = img_pil.size
    escala = min(ancho_max / ancho_orig, alto_max / alto_orig)
    nuevo_ancho = max(1, int(ancho_orig * escala))
    nuevo_alto = max(1, int(alto_orig * escala))

    # Redimensionamos solo para la visualización en la pantalla
    img_redimensionada = img_pil.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img_redimensionada)


def formatear_mascara_a_texto(mascara):
    """
    Convierte una matriz 2D en un texto tabulado con 4 decimales.
    """
    filas, columnas = mascara.shape
    lineas = []
    for i in range(filas):
        valores = [f"{mascara[i, j]:7.4f}" for j in range(columnas)]
        lineas.append("  [ " + "  ".join(valores) + " ]")
    return "\n".join(lineas)


# =============================================================================
# EVENTOS Y ACCIONES DE LA INTERFAZ
# =============================================================================

def accion_cargar_imagen(ruta_especifica=None):
    """Abre el explorador de archivos para cargar una imagen."""
    global imagen_original, imagen_procesada, imagen_directa, foto_original_tk

    if ruta_especifica is None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
        )
    else:
        ruta = ruta_especifica

    if not ruta or not os.path.exists(ruta):
        return

    try:
        # 1. Leemos la imagen y la convertimos a matriz
        imagen_original = procesamiento.cargar_imagen(ruta)
        imagen_procesada = None
        imagen_directa = None

        alto, ancho, _ = imagen_original.shape
        nombre_archivo = os.path.basename(ruta)

        # Actualizamos etiquetas informativas
        lbl_info_original.config(
            text=f"Archivo: {nombre_archivo}\nDimensiones: {alto} filas x {ancho} columnas (Píxeles: {alto * ancho:,})"
        )
        lbl_info_procesada.config(text="Esperando procesamiento...")

        # Mostramos la imagen original en su visor
        foto_original_tk = convertir_matriz_a_foto_tk(imagen_original)
        panel_imagen_original.config(image=foto_original_tk, text="")
        panel_imagen_procesada.config(image="", text="[ Presione 'Procesar y Reducir' ]")

        btn_procesar.config(state="normal")
        btn_guardar.config(state="disabled")
        btn_comparar.config(state="disabled")

    except Exception as error:
        messagebox.showerror("Error al Cargar", f"No se pudo cargar la imagen:\n{error}")


def accion_actualizar_mascara(*args):
    """
    Genera la matriz de la máscara según las opciones elegidas
    (Promedio, Gaussiana o Personalizada) y la muestra en la pantalla.
    """
    global mascara_actual

    try:
        tamano = int(combo_tamano.get())
    except ValueError:
        tamano = 3

    # El tamaño de la máscara debe ser impar
    if tamano % 2 == 0:
        tamano += 1
        combo_tamano.set(str(tamano))

    tipo = combo_tipo_mascara.get()

    if "Promedio" in tipo:
        mascara_actual = procesamiento.crear_mascara_promedio(tamano)
        txt_mascara.config(state="normal")
        txt_mascara.delete("1.0", tk.END)
        txt_mascara.insert(tk.END, formatear_mascara_a_texto(mascara_actual))
        txt_mascara.config(state="disabled")
        lbl_nota_mascara.config(text=f"Filtro Promedio: Cada celda vale 1/{tamano*tamano} = {1.0/(tamano*tamano):.4f}. Suma = 1.0")

    elif "Gaussiana" in tipo:
        try:
            sigma = float(entry_sigma.get())
        except ValueError:
            sigma = 1.0
            entry_sigma.delete(0, tk.END)
            entry_sigma.insert(0, "1.0")

        mascara_actual = procesamiento.crear_mascara_gaussiana(tamano, sigma)
        txt_mascara.config(state="normal")
        txt_mascara.delete("1.0", tk.END)
        txt_mascara.insert(tk.END, formatear_mascara_a_texto(mascara_actual))
        txt_mascara.config(state="disabled")
        lbl_nota_mascara.config(text=f"Filtro Gaussiano (sigma={sigma}): Mayor peso central, atenuación radial. Suma = 1.0")

    else:  # Personalizada
        txt_mascara.config(state="normal")
        lbl_nota_mascara.config(text="Máscara Personalizada: Puede editar los valores numéricos y pulsar 'Aplicar Máscara'.")


def accion_aplicar_mascara_personalizada():
    """Lee los números ingresados por el usuario en el cuadro de texto y los normaliza."""
    global mascara_actual
    try:
        texto = txt_mascara.get("1.0", tk.END).strip()
        lineas = texto.split("\n")
        filas = []
        for linea in lineas:
            linea_limpia = linea.replace("[", "").replace("]", "").replace(",", " ").strip()
            if linea_limpia:
                valores = [float(val) for val in linea_limpia.split()]
                filas.append(valores)

        matriz = np.array(filas, dtype=np.float64)
        if matriz.shape[0] != matriz.shape[1]:
            messagebox.showwarning("Atención", "La máscara debe ser una matriz cuadrada (NxN).")
            return

        mascara_actual = procesamiento.normalizar_mascara(matriz)
        txt_mascara.delete("1.0", tk.END)
        txt_mascara.insert(tk.END, formatear_mascara_a_texto(mascara_actual))
        lbl_nota_mascara.config(text=f"Máscara Personalizada de {mascara_actual.shape[0]}x{mascara_actual.shape[1]} normalizada (Suma = 1.0).")
        messagebox.showinfo("Éxito", "Máscara personalizada cargada y normalizada correctamente.")
    except Exception as error:
        messagebox.showerror("Error", f"Formato inválido de máscara:\n{error}")


def accion_procesar_imagen():
    """Ejecuta la reducción con suavizado y la reducción directa."""
    global imagen_original, imagen_procesada, imagen_directa, mascara_actual, foto_procesada_tk

    if imagen_original is None:
        messagebox.showwarning("Aviso", "Primero debe cargar una imagen.")
        return

    try:
        # Obtenemos los saltos vertical y horizontal
        salto_fila = int(combo_salto_v.get())
        salto_columna = int(combo_salto_h.get())
    except ValueError:
        messagebox.showwarning("Aviso", "Los saltos deben ser números enteros válidos.")
        return

    # Si no hay máscara generada, la actualizamos
    if mascara_actual is None:
        accion_actualizar_mascara()

    # Mostramos estado en interfaz
    lbl_info_procesada.config(text="Procesando convolución y reducción... Por favor espere.")
    ventana_principal.update()

    # 1. Reducción con suavizado (Convolución + saltos)
    imagen_procesada = procesamiento.reducir_con_suavizado(
        imagen_original, mascara_actual, salto_fila, salto_columna
    )

    # 2. Reducción directa sin suavizado (para fines comparativos)
    imagen_directa = procesamiento.reducir_directo_sin_suavizado(
        imagen_original, salto_fila, salto_columna
    )

    # Obtenemos dimensiones finales
    alto_orig, ancho_orig, _ = imagen_original.shape
    alto_fin, ancho_fin, _ = imagen_procesada.shape
    porcentaje_filas = (alto_fin / alto_orig) * 100
    porcentaje_cols = (ancho_fin / ancho_orig) * 100
    porcentaje_total = ((alto_fin * ancho_fin) / (alto_orig * ancho_orig)) * 100

    lbl_info_procesada.config(
        text=(
            f"Dimensiones finales: {alto_fin} filas x {ancho_fin} columnas\n"
            f"Escala: {porcentaje_cols:.1f}% ancho x {porcentaje_filas:.1f}% alto | Píxeles totales: {alto_fin * ancho_fin:,} ({porcentaje_total:.1f}% del original)"
        )
    )

    # Actualizamos el visor con la imagen reducida
    foto_procesada_tk = convertir_matriz_a_foto_tk(imagen_procesada)
    panel_imagen_procesada.config(image=foto_procesada_tk, text="")

    btn_guardar.config(state="normal")
    btn_comparar.config(state="normal")


def accion_guardar_imagen():
    """Guarda la imagen procesada en disco."""
    global imagen_procesada
    if imagen_procesada is None:
        return

    ruta_destino = filedialog.asksaveasfilename(
        title="Guardar Imagen Reducida",
        defaultextension=".jpg",
        filetypes=[("Imagen JPEG", "*.jpg"), ("Imagen PNG", "*.png")]
    )
    if ruta_destino:
        try:
            procesamiento.guardar_imagen(imagen_procesada, ruta_destino)
            messagebox.showinfo("Guardado", f"Imagen guardada con éxito en:\n{ruta_destino}")
        except Exception as error:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar la imagen:\n{error}")


def accion_mostrar_comparacion():
    """
    Abre una ventana secundaria que compara lado a lado la reducción
    CON suavizado vs la reducción DIRECTA (sin suavizado), mostrando la diferencia matemática.
    """
    global imagen_procesada, imagen_directa, foto_comp_suavizada_tk, foto_comp_directa_tk

    if imagen_procesada is None or imagen_directa is None:
        messagebox.showwarning("Aviso", "Primero debe procesar la imagen.")
        return

    # Ventana modal de comparación
    ventana_comp = tk.Toplevel(ventana_principal)
    ventana_comp.title("Comparación: Reducción con Suavizado vs Reducción Directa")
    ventana_comp.geometry("860x520")
    ventana_comp.minsize(800, 500)

    # Cálculo de la diferencia promedio
    diferencia_media = procesamiento.calcular_diferencia_absoluta(imagen_procesada, imagen_directa)

    frame_cabecera = ttk.Frame(ventana_comp, padding=10)
    frame_cabecera.pack(fill="x")

    lbl_titulo_comp = ttk.Label(
        frame_cabecera,
        text="Efecto del Suavizado Previo (Filtro Anti-Aliasing)",
        font=("Arial", 12, "bold")
    )
    lbl_titulo_comp.pack(anchor="w")

    texto_explicativo = (
        f"Diferencia absoluta promedio: {diferencia_media:.2f} niveles de intensidad.\n"
        "• Reducción DIRECTA (Derecha): Salta píxeles sin considerar los vecinos, produciendo bordes dentados y ruido.\n"
        "• Reducción CON SUAVIZADO (Izquierda): Integra la información de los vecinos mediante convolución, conservando suavidad."
    )
    lbl_desc_comp = ttk.Label(frame_cabecera, text=texto_explicativo, justify="left")
    lbl_desc_comp.pack(anchor="w", pady=4)

    # Contenedor para ambas imágenes lado a lado
    frame_fotos = ttk.Frame(ventana_comp, padding=10)
    frame_fotos.pack(fill="both", expand=True)

    # 1. Columna izquierda: Con suavizado
    col_suavizada = ttk.LabelFrame(frame_fotos, text=" 1. Reducción CON Suavizado (Filtro) ", padding=8)
    col_suavizada.pack(side="left", fill="both", expand=True, padx=5)

    foto_comp_suavizada_tk = convertir_matriz_a_foto_tk(imagen_procesada, ancho_max=390, alto_max=330)
    lbl_foto_suavizada = ttk.Label(col_suavizada, image=foto_comp_suavizada_tk, anchor="center")
    lbl_foto_suavizada.pack(fill="both", expand=True)

    # 2. Columna derecha: Directa sin suavizado
    col_directa = ttk.LabelFrame(frame_fotos, text=" 2. Reducción DIRECTA (Sin Suavizado) ", padding=8)
    col_directa.pack(side="right", fill="both", expand=True, padx=5)

    foto_comp_directa_tk = convertir_matriz_a_foto_tk(imagen_directa, ancho_max=390, alto_max=330)
    lbl_foto_directa = ttk.Label(col_directa, image=foto_comp_directa_tk, anchor="center")
    lbl_foto_directa.pack(fill="both", expand=True)

    btn_cerrar = ttk.Button(ventana_comp, text="Cerrar Comparación", command=ventana_comp.destroy)
    btn_cerrar.pack(pady=8)


# =============================================================================
# CONSTRUCCIÓN DE LA INTERFAZ PRINCIPAL
# =============================================================================

ventana_principal = tk.Tk()
ventana_principal.title("Taller: Reducción y Suavizado de Imágenes con Máscara")
ventana_principal.geometry("1100x680")
ventana_principal.minsize(980, 620)

# Estilo general
estilo = ttk.Style()
estilo.theme_use("clam")

# Contenedor dividido en 2 columnas: Controles a la izquierda, Imágenes a la derecha
panel_izquierdo = ttk.Frame(ventana_principal, padding=12, width=340)
panel_izquierdo.pack(side="left", fill="y")

panel_derecho = ttk.Frame(ventana_principal, padding=12)
panel_derecho.pack(side="right", fill="both", expand=True)


# -----------------------------------------------------------------------------
# PANEL IZQUIERDO: CONTROLES
# -----------------------------------------------------------------------------

# Grupo 1: Carga de imagen
grupo_carga = ttk.LabelFrame(panel_izquierdo, text=" 1. Imagen de Entrada ", padding=8)
grupo_carga.pack(fill="x", pady=4)

btn_cargar = ttk.Button(grupo_carga, text="Cargar Imagen desde Archivo", command=accion_cargar_imagen)
btn_cargar.pack(fill="x", pady=3)

btn_muestra = ttk.Button(
    grupo_carga,
    text="Cargar Paisaje de Muestra",
    command=lambda: accion_cargar_imagen("imagenes/paisaje.jpg")
)
btn_muestra.pack(fill="x", pady=2)


# Grupo 2: Configuración de la máscara
grupo_mascara = ttk.LabelFrame(panel_izquierdo, text=" 2. Máscara de Suavizado ", padding=8)
grupo_mascara.pack(fill="x", pady=6)

ttk.Label(grupo_mascara, text="Tipo de máscara:").pack(anchor="w")
combo_tipo_mascara = ttk.Combobox(
    grupo_mascara,
    values=["Promedio (Box Filter)", "Gaussiana", "Personalizada"],
    state="readonly"
)
combo_tipo_mascara.set("Promedio (Box Filter)")
combo_tipo_mascara.pack(fill="x", pady=3)
combo_tipo_mascara.bind("<<ComboboxSelected>>", accion_actualizar_mascara)

frame_tam_sigma = ttk.Frame(grupo_mascara)
frame_tam_sigma.pack(fill="x", pady=3)

ttk.Label(frame_tam_sigma, text="Tamaño:").grid(row=0, column=0, sticky="w", padx=2)
combo_tamano = ttk.Combobox(frame_tam_sigma, values=["3", "5", "7"], width=5, state="readonly")
combo_tamano.set("3")
combo_tamano.grid(row=0, column=1, sticky="w", padx=4)
combo_tamano.bind("<<ComboboxSelected>>", accion_actualizar_mascara)

ttk.Label(frame_tam_sigma, text="Sigma (Gaus):").grid(row=0, column=2, sticky="w", padx=6)
entry_sigma = ttk.Entry(frame_tam_sigma, width=6)
entry_sigma.insert(0, "1.0")
entry_sigma.grid(row=0, column=3, sticky="w")
entry_sigma.bind("<Return>", accion_actualizar_mascara)

# Visualización de la matriz de la máscara
ttk.Label(grupo_mascara, text="Matriz de la Máscara (Pesos normalizados):").pack(anchor="w", pady=(6, 2))

txt_mascara = tk.Text(grupo_mascara, height=5, width=34, font=("Consolas", 8), bg="#f8f9fa", relief="solid", bd=1)
txt_mascara.pack(fill="x", pady=2)

lbl_nota_mascara = ttk.Label(grupo_mascara, text="", font=("Arial", 7), wraplength=280)
lbl_nota_mascara.pack(anchor="w", pady=2)

btn_aplicar_pers = ttk.Button(grupo_mascara, text="Aplicar Valores Editados (Personalizada)", command=accion_aplicar_mascara_personalizada)
btn_aplicar_pers.pack(fill="x", pady=2)


# Grupo 3: Parámetros de salto (Strides)
grupo_saltos = ttk.LabelFrame(panel_izquierdo, text=" 3. Reducción / Saltos (Strides) ", padding=8)
grupo_saltos.pack(fill="x", pady=6)

frame_saltos = ttk.Frame(grupo_saltos)
frame_saltos.pack(fill="x")

ttk.Label(frame_saltos, text="Salto Vertical (Filas):").grid(row=0, column=0, sticky="w", pady=3)
combo_salto_v = ttk.Combobox(frame_saltos, values=["1", "2", "3", "4", "5"], width=4, state="readonly")
combo_salto_v.set("2")
combo_salto_v.grid(row=0, column=1, padx=8, pady=3)

ttk.Label(frame_saltos, text="Salto Horizontal (Cols):").grid(row=1, column=0, sticky="w", pady=3)
combo_salto_h = ttk.Combobox(frame_saltos, values=["1", "2", "3", "4", "5"], width=4, state="readonly")
combo_salto_h.set("2")
combo_salto_h.grid(row=1, column=1, padx=8, pady=3)

ttk.Label(grupo_saltos, text="* Salto 2x2 reduce las dimensiones al 50% (1/4 de píxeles).", font=("Arial", 7)).pack(anchor="w", pady=2)


# Grupo 4: Acciones principales
grupo_acciones = ttk.LabelFrame(panel_izquierdo, text=" 4. Ejecución ", padding=8)
grupo_acciones.pack(fill="x", pady=6)

btn_procesar = ttk.Button(grupo_acciones, text="▶ Procesar y Reducir Imagen", command=accion_procesar_imagen, state="disabled")
btn_procesar.pack(fill="x", pady=3)

btn_comparar = ttk.Button(grupo_acciones, text="⚖ Comparar: Suavizado vs Directa", command=accion_mostrar_comparacion, state="disabled")
btn_comparar.pack(fill="x", pady=3)

btn_guardar = ttk.Button(grupo_acciones, text="💾 Guardar Imagen Reducida", command=accion_guardar_imagen, state="disabled")
btn_guardar.pack(fill="x", pady=3)


# -----------------------------------------------------------------------------
# PANEL DERECHO: VISORES DE IMÁGENES
# -----------------------------------------------------------------------------

# Contenedor para las dos imágenes lado a lado
frame_visores = ttk.Frame(panel_derecho)
frame_visores.pack(fill="both", expand=True)

# Visor 1: Imagen Original
col_original = ttk.LabelFrame(frame_visores, text=" Imagen Original ", padding=8)
col_original.pack(side="left", fill="both", expand=True, padx=4)

lbl_info_original = ttk.Label(col_original, text="Ninguna imagen cargada", justify="center")
lbl_info_original.pack(fill="x", pady=4)

panel_imagen_original = ttk.Label(col_original, text="[ Sin Imagen ]", anchor="center", background="#e9ecef")
panel_imagen_original.pack(fill="both", expand=True)


# Visor 2: Imagen Procesada
col_procesada = ttk.LabelFrame(frame_visores, text=" Imagen Reducida con Suavizado ", padding=8)
col_procesada.pack(side="right", fill="both", expand=True, padx=4)

lbl_info_procesada = ttk.Label(col_procesada, text="Esperando procesamiento...", justify="center")
lbl_info_procesada.pack(fill="x", pady=4)

panel_imagen_procesada = ttk.Label(col_procesada, text="[ Sin Imagen ]", anchor="center", background="#e9ecef")
panel_imagen_procesada.pack(fill="both", expand=True)


# =============================================================================
# INICIALIZACIÓN AUTOMÁTICA
# =============================================================================

# Inicializamos la máscara por defecto
accion_actualizar_mascara()

# Si existe la imagen de muestra 'imagenes/paisaje.jpg', la cargamos automáticamente
if os.path.exists("imagenes/paisaje.jpg"):
    accion_cargar_imagen("imagenes/paisaje.jpg")


if __name__ == "__main__":
    ventana_principal.mainloop()
