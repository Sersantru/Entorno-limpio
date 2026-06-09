#!/usr/bin/env python3
from pathlib import Path
import cv2
from flask import Flask, jsonify, render_template
from ultralytics import YOLO
import threading
import time
import queue
import easyocr

# --- PARCHE DE COMPATIBILIDAD NUMPY ---
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

app = Flask(__name__)

CAMERA_URL = "http://192.168.150.244:8123/video"
# MODEL_PATH = Path(__file__).resolve().parent / "modelo-detector-matricula.pt"
MODEL_PATH = Path(__file__).resolve().parent / "modelo-detector-matricula.engine"

lock = threading.Lock()
frame_actual = None
coordenadas_guardadas = []

# Almacén de resultados del OCR: {track_id: "MATRÍCULA_DETECTADA"}
resultados_ocr = {}
# Historial para no machacar el OCR mil veces con el mismo coche
ids_en_proceso_o_leidos = set()

# Cola interna para pasar los recortes al hilo de OCR de forma síncrona y segura
cola_ocr = queue.Queue(maxsize=10)

def bucle_lectura_camara():
    """Hilo que lee la cámara a la máxima velocidad posible para reventar el buffer"""
    global frame_actual
    print("[CÁMARA] Hilo de captura en tiempo real iniciado.")

    cap = cv2.VideoCapture(CAMERA_URL)

    while True:
        if not cap.isOpened():
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        ok, frame = cap.read()
        if not ok:
            print("[CÁMARA] Stream caído, reiniciando captura...")
            cap.release()
            time.sleep(0.15)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        with lock:
            frame_actual = frame

def bucle_inferencia():
    """Hilo dedicado a inferencia YOLO sobre la GPU"""
    global coordenadas_guardadas, frame_actual

    print("[IA] Cargando motor TensorRT en el hilo dedicado...")
    model = YOLO(str(MODEL_PATH), task='detect')
    print("[IA] Motor de inferencia arrancado.")

    while True:
        try:
            with lock:
                if frame_actual is None:
                    frame_ia = None
                else:
                    frame_ia = frame_actual.copy()

            if frame_ia is not None:
                results = model.track(frame_ia, verbose=False, conf=0.30, persist=True)

                nuevas_cajas = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        track_id = int(box.id[0].item()) if box.id is not None else None

                        # Si hay un ID válido y no lo hemos procesado ya,mandamos el recorte al OCR
                        if track_id is not None and track_id not in ids_en_proceso_o_leidos:
                            # Recorte directo sobre el array de NumPy (rápido y en memoria)
                            recorte = frame_ia[int(y1):int(y2), int(x1):int(x2)]
                            
                            if recorte.size > 0:
                                try:
                                    # Metemos a la cola el ID y su recorte sin bloquear el bucle de la IA
                                    cola_ocr.put_nowait((track_id, recorte))
                                    ids_en_proceso_o_leidos.add(track_id)
                                    print(f"[IA] Matrícula con ID {track_id} enviada a la cola de OCR.")
                                except queue.Full:
                                    # Si la cola está llena, ignoramos este frame para no ralentizar la GPU
                                    pass

                        # Buscamos si este ID ya tiene texto procesado por el OCR para meterlo en el JSON
                        texto_ocr = resultados_ocr.get(track_id, "")


                        nuevas_cajas.append([int(x1), int(y1), int(x2), int(y2), conf, track_id, texto_ocr])

                with lock:
                    coordenadas_guardadas = nuevas_cajas
            else:
                time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR CRÍTICO EN IA] El hilo de inferencia ha reventado: {e}")
            time.sleep(1)


def bucle_ocr():
    """Hilo 3: Consumidor de recortes para aplicar el OCR sin penalizar los FPS de la IA"""
    global resultados_ocr
    print("[OCR] Cargando modelo de OCR en memoria...")
    
    # --- AQUÍ INICIALIZAS TU LIBRERÍA DE OCR ---
    reader = easyocr.Reader(['es'])
    print("[OCR] Modelo de OCR listo para procesar.")

    while True:
        try:
            # Este método se queda esperando (bloqueado sin consumir CPU) hasta que entre un recorte
            track_id, recorte_matricula = cola_ocr.get()
            
            print(f"[OCR] Procesando lectura para el ID: {track_id}...")
            
            # --- AQUÍ EJECUTAS TU LECTURA ---
            # Ejemplo ficticio:
            resultado = reader.readtext(recorte_matricula)
            
            if resultado:
                # EasyOCR devuelve una lista de tuplas: [(bbox, texto, confianza), ...]
                texto_detectado = resultado[0][1].strip().upper()
            else:
                texto_detectado = "LEYENDO..."
            
            # Guardamos el resultado asociado a su ID de seguimiento
            with lock:
                resultados_ocr[track_id] = texto_detectado
            
            print(f"[OCR] Éxito ID {track_id} -> {texto_detectado}")
            cola_ocr.task_done()

        except Exception as e:
            print(f"[ERROR EN OCURRENCIA OCR]: {e}")
            time.sleep(0.5)

@app.route('/')
def index():
    return render_template('index.html', camera_url=CAMERA_URL)

@app.route('/coordenadas')
def coordenadas():
    with lock:
        cajas_actuales = coordenadas_guardadas

    return jsonify(cajas_actuales)

if __name__ == "__main__":
    hilo_cam = threading.Thread(target=bucle_lectura_camara, daemon=True)
    hilo_cam.start()

    hilo_ia = threading.Thread(target=bucle_inferencia, daemon=True)
    hilo_ia.start()

    hilo_ocr = threading.Thread(target=bucle_ocr, daemon=True)
    hilo_ocr.start()

    app.run(host='0.0.0.0', port=8182, threaded=True)
