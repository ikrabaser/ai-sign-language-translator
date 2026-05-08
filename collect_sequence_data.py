import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import sys

DATA_PATH = "sequence_data"
SEQUENCE_LENGTH = 30
DATA_COUNT = 30

if len(sys.argv) > 1:
    etiket = sys.argv[1]
else:
    etiket = input("Etiket gir: ")

etiket_klasoru = os.path.join(DATA_PATH, etiket)
os.makedirs(etiket_klasoru, exist_ok=True)

kamera = cv2.VideoCapture(0)

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

eller = mp_eller.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

kayit_no = len(os.listdir(etiket_klasoru))

while kayit_no < DATA_COUNT:
    kareler = []

    for kare_no in range(SEQUENCE_LENGTH):
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

        kareler.append(koordinatlar)

        cv2.putText(
            goruntu,
            f"Etiket: {etiket}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            goruntu,
            f"Kayit: {kayit_no + 1}/{DATA_COUNT} | Kare: {kare_no + 1}/{SEQUENCE_LENGTH}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Hareketli Veri Toplama", goruntu)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            kamera.release()
            cv2.destroyAllWindows()
            exit()

    if len(kareler) == SEQUENCE_LENGTH:
        dosya_yolu = os.path.join(etiket_klasoru, f"{kayit_no}.npy")
        np.save(dosya_yolu, np.array(kareler))
        print(f"Kaydedildi: {dosya_yolu}")
        kayit_no += 1

kamera.release()
cv2.destroyAllWindows()

print("Veri toplama tamamlandı.")