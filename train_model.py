import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

veri = pd.read_csv("hand_data.csv", header=None)

X = veri.iloc[:, :-1]
y = veri.iloc[:, -1]

etiket_cevirici = LabelEncoder()
y_encoded = etiket_cevirici.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

basari = model.score(X_test, y_test)

print("Model başarı oranı:", basari)

with open("sign_model.pkl", "wb") as dosya:
    pickle.dump(model, dosya)

with open("label_encoder.pkl", "wb") as dosya:
    pickle.dump(etiket_cevirici, dosya)

print("Model kaydedildi.")
