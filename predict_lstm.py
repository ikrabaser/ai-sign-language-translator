import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
from tensorflow.keras.models import load_model

model = load_model("lstm_sign_model.h5")

with open("lstm_label_encoder.pkl", "rb") as dosya:
    etiket_cevirici = pickle.load(dosya)

kamera = cv2.VideoCapture(0)

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

eller = mp_eller.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

sequence = deque(maxlen=30)

tahmin_etiketi = ""

while True:
    basarili, goruntu = kamera.read()

    if not basarili:
        break

    goruntu = cv2.flip(goruntu, 1)

    rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)

    sonuc = eller.process(rgb)

    koordinatlar = np.zeros(63)

    if sonuc.multi_hand_landmarks:
        el = sonuc.multi_hand_landmarks[0]

        koordinatlar = []

        for nokta in el.landmark:
            koordinatlar.extend([nokta.x, nokta.y, nokta.z])

        koordinatlar = np.array(koordinatlar)

        mp_cizim.draw_landmarks(
            goruntu,
            el,
            mp_eller.HAND_CONNECTIONS
        )

    sequence.append(koordinatlar)

    if len(sequence) == 30:
        veri = np.expand_dims(sequence, axis=0)

        tahmin = model.predict(veri, verbose=0)

        tahmin_index = np.argmax(tahmin)

        tahmin_etiketi = etiket_cevirici.inverse_transform(
            [tahmin_index]
        )[0]

    cv2.putText(
        goruntu,
        f"Tahmin: {tahmin_etiketi}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("LSTM Isaret Dili Tahmini", goruntu)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()