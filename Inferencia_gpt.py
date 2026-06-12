#!/usr/bin/env python3
from pathlib import Path
import cv2
from flask import Flask, jsonify, render_template
from ultralytics import YOLO
from datetime import datetime
import threading
import time
import numpy as np

# --- COMPAT NUMPY ---
if not hasattr(np, 'bool'):
    np.bool = np.bool_

app = Flask(__name__)

CAMERA_URL = "http://192.168.150.244:8123/video"
MODEL_PATH = Path(__file__).resolve().parent / "modelo-detector-matricula.engine"

# ------------------------
# CONFIG
# ------------------------
CONF_MINIMA_HTML = 0.40
TIEMPO_EXPIRACION = 3.0  # segundos sin verse para cerrar vehículo

# ------------------------
# GLOBAL STATE
# ------------------------
lock = threading.Lock()

frame_actual = None
coordenadas_guardadas = []

vehiculos = {}  # track_id -> estado


# =========================================================
# CAMARA
# =========================================================
def bucle_lectura_camara():
    global frame_actual

    cap = cv2.VideoCapture(CAMERA_URL)

    print("[CÁMARA] Iniciado")

    while True:
        if not cap.isOpened():
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        ok, frame = cap.read()

        if not ok:
            cap.release()
            time.sleep(0.2)
            cap = cv2.VideoCapture(CAMERA_URL)
            continue

        with lock:
            frame_actual = frame


# =========================================================
# IA + TRACKING + OCR QUEUE LOGIC SIMPLE
# =========================================================
def bucle_inferencia():
    global coordenadas_guardadas, frame_actual, vehiculos

    print("[IA] Cargando modelo TensorRT...")
    model = YOLO(str(MODEL_PATH), task="detect")
    print("[IA] OK")

    while True:

        try:
            with lock:
                if frame_actual is None:
                    continue
                frame = frame_actual.copy()

            results = model.track(frame, verbose=False, conf=0.30, persist=True)

            nuevas_cajas = []

            now = time.time()

            for result in results:
                for box in result.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    track_id = int(box.id[0].item()) if box.id is not None else None

                    if track_id is None:
                        continue

                    # clamp coords
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    # init state
                    if track_id not in vehiculos:
                        vehiculos[track_id] = {
                            "texto": "",
                            "confianza": 0.0,
                            "ultimo_visto": now,
                            "escrito": False,
                            "ultimo_bbox": (x1, y1, x2, y2),
                            "area_max": 0
                        }

                    v = vehiculos[track_id]

                    # update last seen
                    v["ultimo_visto"] = now
                    v["ultimo_bbox"] = (x1, y1, x2, y2)

                    area = (x2 - x1) * (y2 - y1)

                    # guardar mejor bbox (sin recortes en RAM)
                    if area > v["area_max"]:
                        v["area_max"] = area

                    texto = v["texto"] if v["confianza"] >= CONF_MINIMA_HTML else ""

                    nuevas_cajas.append([x1, y1, x2, y2, conf, track_id, texto])

            with lock:
                coordenadas_guardadas = nuevas_cajas

        except Exception as e:
            print("[ERROR IA]", e)
            time.sleep(0.5)


# =========================================================
# OCR (SE EJECUTA SOLO CUANDO VEHÍCULO DESAPARECE)
# =========================================================
def bucle_ocr():
    import easyocr

    print("[OCR] Cargando modelo...")
    reader = easyocr.Reader(['es'])
    print("[OCR] listo")

    while True:

        try:
            time.sleep(0.5)

            now = time.time()

            ids_to_delete = []

            with lock:

                for track_id, v in vehiculos.items():

                    # vehículo desaparecido
                    if now - v["ultimo_visto"] > TIEMPO_EXPIRACION:

                        if not v["escrito"]:

                            x1, y1, x2, y2 = v["ultimo_bbox"]

                            with lock:
                                frame = frame_actual.copy() if frame_actual is not None else None

                            if frame is None:
                                continue

                            recorte = frame[y1:y2, x1:x2]

                            if recorte.size == 0:
                                continue

                            # OCR
                            resultado = reader.readtext(recorte)

                            if resultado:
                                mejor = max(resultado, key=lambda x: x[2])
                                texto = mejor[1].strip().upper()
                                conf = float(mejor[2])

                                # guardar si es mejor
                                if conf > v["confianza"]:
                                    v["texto"] = texto
                                    v["confianza"] = conf

                        # escribir al TXT solo si pasa umbral
                        if v["confianza"] >= CONF_MINIMA_HTML and not v["escrito"]:

                            hora = datetime.now().strftime("%H:%M:%S")

                            with open("resultados.txt", "a") as f:
                                f.write(
                                    f"{hora};{track_id};{v['texto']};{v['confianza']:.2f}\n"
                                )

                            print(f"[TXT] Guardado {track_id} -> {v['texto']}")

                            v["escrito"] = True

                        ids_to_delete.append(track_id)

            # limpieza fuera del lock
            with lock:
                for i in ids_to_delete:
                    del vehiculos[i]

        except Exception as e:
            print("[ERROR OCR]", e)
            time.sleep(1)


# =========================================================
# FLASK
# =========================================================
@app.route('/')
def index():
    return render_template('index.html', camera_url=CAMERA_URL)


@app.route('/coordenadas')
def coordenadas():
    with lock:
        return jsonify(coordenadas_guardadas)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    threading.Thread(target=bucle_lectura_camara, daemon=True).start()
    threading.Thread(target=bucle_inferencia, daemon=True).start()
    threading.Thread(target=bucle_ocr, daemon=True).start()

    app.run(host='0.0.0.0', port=8182, threaded=True)