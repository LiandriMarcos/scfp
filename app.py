from flask import Flask, jsonify, send_from_directory
import numpy as np
import cv2
from threading import Thread
import time

app = Flask(__name__)

# Variáveis globais para armazenar os contadores
total = 0
up = 0
down = 0

# Variáveis para manter o snapshot atualizado dos contadores
cont_snapshot = 0
up_snapshot = 0
down_snapshot = 0

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

@app.route('/api/contagem', methods=['GET'])
def get_contagem():
    global cont_snapshot, up_snapshot, down_snapshot
    return jsonify({'cont': cont_snapshot, 'subindo': up_snapshot, 'descendo': down_snapshot})

@app.route('/')
def serve_frontend():
    return send_from_directory('', 'cont.html')

def process_video():
    global total, up, down

    cap = cv2.VideoCapture('http://192.168.255.106:4747/video')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    detects = []

    posL = 250
    offset = 40
    xy1 = (0, posL)
    xy2 = (700, posL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        dilation = cv2.dilate(opening, kernel, iterations=8)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)

        cv2.line(frame, xy1, xy2, (255, 0, 0), 3)
        cv2.line(frame, (xy1[0], posL - offset), (xy2[0], posL - offset), (255, 255, 0), 2)
        cv2.line(frame, (xy1[0], posL + offset), (xy2[0], posL + offset), (255, 255, 0), 2)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            if int(area) > 3000:
                centro = center(x, y, w, h)

                cv2.putText(frame, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, centro, 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if len(detects) <= i:
                    detects.append([])
                if centro[1] > posL - offset and centro[1] < posL + offset:
                    detects[i].append(centro)
                else:
                    detects[i].clear()
                i += 1

        if i == 0:
            detects.clear()

        if len(contours) == 0:
            detects.clear()
        else:
            for detect in detects:
                for (c, l) in enumerate(detect):
                    if detect[c - 1][1] < posL and l[1] > posL:
                        detect.clear()
                        up += 1
                        total += 1
                        cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                        continue

                    if detect[c - 1][1] > posL and l[1] < posL:
                        detect.clear()
                        down += 1
                        total += 1
                        cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                        continue

                    if c > 0:
                        cv2.line(frame, detect[c - 1], l, (0, 0, 255), 1)

        cv2.putText(frame, "SUBINDO: " + str(up), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "DESCENDO: " + str(down), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "CONT: " + str(up - down), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_api_data():
    global cont_snapshot, up_snapshot, down_snapshot, up, down
    while True:
        time.sleep(1)
        cont_snapshot = up - down
        up_snapshot = up
        down_snapshot = down
        print(f"Contagem: {cont_snapshot}, Subindo: {up_snapshot}, Descendo: {down_snapshot}")  # Adicione um log para depuração


if __name__ == '__main__':
    video_thread = Thread(target=process_video)
    video_thread.start()

    api_update_thread = Thread(target=update_api_data)
    api_update_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
