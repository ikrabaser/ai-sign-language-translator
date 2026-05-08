import cv2
import mediapipe as mp

kamera = cv2.VideoCapture(0)

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

eller = mp_eller.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

while True:
    basarili, goruntu = kamera.read()

    if not basarili:
        print("Kamera görüntüsü alınamadı.")
        break

    goruntu = cv2.flip(goruntu, 1)

    rgb_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    sonuc = eller.process(rgb_goruntu)

    if sonuc.multi_hand_landmarks:
        for el_noktalari in sonuc.multi_hand_landmarks:
            mp_cizim.draw_landmarks(
                goruntu,
                el_noktalari,
                mp_eller.HAND_CONNECTIONS
            )

    cv2.imshow("El Takibi", goruntu)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()