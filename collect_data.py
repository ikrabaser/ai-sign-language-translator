
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import mediapipe as mp
import csv

kamera = cv2.VideoCapture(0)

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

eller = mp_eller.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

if len(sys.argv) > 1:
    etiket = sys.argv[1]
else:
    etiket = input("Etiket gir: ")

print("Kaydedilen etiket:", etiket)

csv_dosyasi = open("hand_data.csv", mode="a", newline="")
csv_yazici = csv.writer(csv_dosyasi)

while True:
    basarili, goruntu = kamera.read()

    if not basarili:
        break

    goruntu = cv2.flip(goruntu, 1)

    rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)

    sonuc = eller.process(rgb)

    if sonuc.multi_hand_landmarks:
        for el in sonuc.multi_hand_landmarks:

            koordinatlar = []

            for nokta in el.landmark:
                koordinatlar.append(nokta.x)
                koordinatlar.append(nokta.y)
                koordinatlar.append(nokta.z)

            koordinatlar.append(etiket)

            csv_yazici.writerow(koordinatlar)

            mp_cizim.draw_landmarks(
                goruntu,
                el,
                mp_eller.HAND_CONNECTIONS
            )

    cv2.imshow("Veri Toplama", goruntu)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

csv_dosyasi.close()
kamera.release()
cv2.destroyAllWindows()