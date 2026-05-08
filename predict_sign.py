import cv2
import mediapipe as mp
import pickle
import numpy as np

with open("sign_model.pkl", "rb") as dosya:
    model = pickle.load(dosya)

with open("label_encoder.pkl", "rb") as dosya:
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

            veri = np.array(koordinatlar).reshape(1, -1)

            tahmin = model.predict(veri)
            etiket = etiket_cevirici.inverse_transform(tahmin)[0]

            cv2.putText(
                goruntu,
                f"Tahmin: {etiket}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            mp_cizim.draw_landmarks(
                goruntu,
                el,
                mp_eller.HAND_CONNECTIONS
            )

    cv2.imshow("Canli Isaret Dili Tahmini", goruntu)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()