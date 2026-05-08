import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

DATA_PATH = "sequence_data"

X = []
y = []

etiketler = os.listdir(DATA_PATH)

for etiket in etiketler:
    klasor_yolu = os.path.join(DATA_PATH, etiket)

    for dosya in os.listdir(klasor_yolu):
        dosya_yolu = os.path.join(klasor_yolu, dosya)

        sequence = np.load(dosya_yolu)

        X.append(sequence)
        y.append(etiket)

X = np.array(X)
y = np.array(y)

etiket_cevirici = LabelEncoder()
y_encoded = etiket_cevirici.fit_transform(y)

y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30, 63)))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))
model.add(Dense(len(etiketler), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

kayip, basari = model.evaluate(X_test, y_test)

print("Test başarı oranı:", basari)

model.save("lstm_sign_model.h5")

with open("lstm_label_encoder.pkl", "wb") as dosya:
    pickle.dump(etiket_cevirici, dosya)

print("LSTM model kaydedildi.")