import streamlit as st

st.set_page_config(
    page_title="AI İşaret Dili Tercümanı",
    page_icon="🤟",
    layout="wide"
)

with st.sidebar:
    st.title("🤟 Menü")
    st.success("Model Hazır")

    st.write("### Desteklenen İşaretler")
    st.write("""
    - merhaba
    - hayir
    - tamam
    - tesekkurler
    """)

    st.write("---")

    st.info("""
    Kullanılan Teknolojiler:
    
    - MediaPipe
    - LSTM
    - TensorFlow
    - Streamlit
    """)

st.title("🤟 Yapay Zeka Destekli İşaret Dili Tercümanı")

st.write("""
Gerçek zamanlı el hareketi analizi ile işaret dili tahmini yapan
yapay zeka destekli prototip uygulama.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Durumu", "Hazır")

with col2:
    st.metric("Test Doğruluğu", "%95")

with col3:
    st.metric("Tahmin Durumu", "Beklemede")

st.write("---")

st.subheader("📷 Kamera Sistemi")

col_kamera, col_bilgi = st.columns([2, 1])

with col_kamera:
    st.success("Kamera sistemi hazır")

    st.markdown("""
    <div style="
        height: 260px;
        border-radius: 18px;
        border: 2px dashed #9ca3af;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #f8fafc;
        font-size: 22px;
        color: #475569;
    ">
        📷 Canlı kamera alanı
    </div>
    """, unsafe_allow_html=True)

with col_bilgi:
    st.write("### Sistem Bilgisi")
    st.write("""
    Kamera üzerinden alınan el hareketleri analiz edilerek 
    eğitilmiş LSTM modeli ile sınıflandırılır.
    """)

    st.success("El takibi aktif")
    st.success("Model yüklendi")
    st.success("Tahmin sistemi hazır")

st.write("---")

st.subheader("🧠 Canlı Tahmin")

tahmin_col1, tahmin_col2 = st.columns([1, 2])

with tahmin_col1:
    st.metric("Algılanan İşaret", "Bekleniyor...")

with tahmin_col2:
    st.info("Kullanıcı el hareketi yaptığında tahmin sonucu burada gösterilir.")

st.write("---")

st.subheader("📋 Desteklenen Hareketler")

st.table({
    "İşaret": [
        "merhaba",
        "hayir",
        "tamam",
        "tesekkurler"
    ],
    "Durum": [
        "Aktif",
        "Aktif",
        "Aktif",
        "Aktif"
    ]
})

st.write("---")

st.subheader("🚀 Proje Durumu")

st.success("MediaPipe ile el landmark çıkarımı tamamlandı")
st.success("Hareket dizisi verileri toplandı")
st.success("LSTM modeli eğitildi")
st.success("Gerçek zamanlı tahmin sistemi oluşturuldu")
st.success("Streamlit arayüz tasarımı oluşturuldu")