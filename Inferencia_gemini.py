#!/usr/bin/env python3
from pathlib import Path
import cv2
from flask import Flask, jsonify, render_template
from ultralytics import YOLO
from datetime import datetime
import time
import multiprocessing
import easyocr

# --- PARCHE DE COMPATIBILIDAD NUMPY ---
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

app = Flask(__name__)

CAMERA_URL = "http://192.168.150.244:8123/video"
MODEL_PATH = Path(__file__).resolve().parent / "modelo-detector-matricula.engine"

# --- VARIABLES GLOBALES PARA FLASK (Solo en el proceso principal) ---
frame_actual = None
coordenadas_guardadas = []
resultados_ocr_dict = {}  # Mantenemos un diccionario en el proceso principal para Flask

# --- COLAS DE MULTIPROCESSING ---
# Importante: multiprocessing.Queue es segura para pasar datos entre procesos
cola_recortes = multiprocessing.Queue(maxsize=20)
cola_resultados = multiprocessing.Queue()

# --- FUNCIONES DE LOS PROCESOS SEPARADOS ---

def proceso_ocr(cola_entrada, cola_salida):
    """
    Este proceso corre completamente aislado.
    Solo se encarga de leer de la cola, procesar OCR y enviar resultados.
    """
    print("[OCR] Proceso OCR iniciado. Cargando EasyOCR...")
    reader = easyocr.Reader(['es'], gpu=False)
    print("[OCR] EasyOCR cargado. Esperando recortes...")

    while True:
        try:
            # Espera a recibir una tupla (id, recorte)
            track_id, recorte = cola_entrada.get()
            
            # --- Lectura OCR ---
            resultado = reader.readtext(recorte)
            
            if resultado:
                texto_detectado = resultado[0][1].strip().upper()
            else:
                texto_detectado = "LEYENDO..."
            
            print(f"[OCR] Éxito ID {track_id} -> {texto_detectado}")
            
            # Envía el resultado a la cola de salida para que lo recoja el escritor/Flask
            cola_salida.put((track_id, texto_detectado))

        except Exception as e:
            print(f"[ERROR OCR] {e}")
            time.sleep(0.5)

def proceso_escritura(cola_resultados_ocr):
    """
    Este proceso lee los resultados del OCR y los guarda a disco.
    ¡SOLO ESCRIBE CUANDO HAY RESULTADOS NUEVOS! (Modo append)
    """
    print("[ESCRITURA] Proceso de escritura iniciado.")
    
    while True:
        try:
            # Espera a recibir un resultado (id, texto)
            track_id, texto = cola_resultados_ocr.get()
            
            # Solo escribimos si hay un texto válido (ignoramos "LEYENDO...")
            if texto != "LEYENDO...":
                 with open("resultados.txt", "a") as f:  # Fíjate: "a" (append), no "w" (write)
                    hora_actual = datetime.now().strftime("%H:%M:%S")
                    f.write(f"{hora_actual}: {track_id}: {texto}\n")
                    print(f"[ESCRITURA] Guardado en disco: {track_id} -> {texto}")

        except Exception as e:
            print(f"[ERROR ESCRITURA] {e}")
            time.sleep(1)


# --- BUCLE PRINCIPAL (IA + Cámara) ---
# Aquí combinamos lectura de cámara e IA para evitar problemas de sincronización excesiva
def bucle_principal(cola_ocr):
    global frame_actual, coordenadas_guardadas, resultados_ocr_dict

    print("[IA] Arrancando bucle principal...")
    model = YOLO(str(MODEL_PATH), task='detect')
    cap = cv2.VideoCapture(CAMERA_URL)
    
    ids_procesados = set()

    while True:
        if not cap.isOpened():
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        ok, frame = cap.read()
        if not ok:
            print("[CÁMARA] Stream caído, reiniciando...")
            cap.release()
            time.sleep(0.1)
            continue
            
        frame_actual = frame # Para Flask si lo quieres mostrar luego

        # --- Inferencia YOLO ---
        results = model.track(frame, verbose=False, conf=0.30, persist=True)
        
        nuevas_cajas = []
        
        # --- Leer resultados del OCR (sin bloquear) ---
        # Leemos la cola de resultados para actualizar Flask lo más rápido posible
        while not cola_resultados.empty():
            try:
                res_id, res_texto = cola_resultados.get_nowait()
                resultados_ocr_dict[res_id] = res_texto
            except multiprocessing.queues.Empty:
                break

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                track_id = int(box.id[0].item()) if box.id is not None else None

                # Si es un ID nuevo, mandamos recorte al proceso OCR
                if track_id is not None and track_id not in ids_procesados:
                    recorte = frame[int(y1):int(y2), int(x1):int(x2)]
                    if recorte.size > 0:
                        try:
                            # Metemos a la cola de multiprocessing
                            cola_ocr.put_nowait((track_id, recorte))
                            ids_procesados.add(track_id)
                            print(f"[IA] ID {track_id} a cola OCR.")
                        except multiprocessing.queues.Full:
                            pass # Cola llena, saltamos
                
                # Obtenemos el texto para pintarlo/enviarlo a Flask
                texto_ocr = resultados_ocr_dict.get(track_id, "")
                nuevas_cajas.append([int(x1), int(y1), int(x2), int(y2), conf, track_id, texto_ocr])

        coordenadas_guardadas = nuevas_cajas

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html', camera_url=CAMERA_URL)

@app.route('/coordenadas')
def coordenadas():
    # Flask lee la variable global del proceso principal
    return jsonify(coordenadas_guardadas)

if __name__ == "__main__":
    # --- ESTA ES LA LÍNEA CRÍTICA QUE FALTABA ---
    # Obliga a crear procesos limpios para que CUDA no se corrompa
    multiprocessing.set_start_method('spawn', force=True)
    # 1. Arrancar el proceso de OCR
    p_ocr = multiprocessing.Process(target=proceso_ocr, args=(cola_recortes, cola_resultados), daemon=True)
    p_ocr.start()

    # 2. Arrancar el proceso de Escritura
    p_escritura = multiprocessing.Process(target=proceso_escritura, args=(cola_resultados,), daemon=True)
    p_escritura.start()

    # 3. El bucle de cámara e IA lo corremos en un hilo dentro del proceso principal
    #    (Flask necesita correr en el hilo principal)
    import threading
    hilo_principal = threading.Thread(target=bucle_principal, args=(cola_recortes,), daemon=True)
    hilo_principal.start()

    # 4. Arrancar Flask
    app.run(host='0.0.0.0', port=8182)