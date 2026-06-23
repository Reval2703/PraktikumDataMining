import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ====================================
# CONFIG
# ====================================
st.set_page_config(
    page_title="Smartphone Addiction Classification",
    page_icon="📱",
    layout="wide"
)

# ====================================
# HEADER
# ====================================
st.title("📱 Klasifikasi Kecanduan Smartphone")
st.caption("Metode Decision Tree")

# ====================================
# LOAD DATASET
# ====================================
df = pd.read_csv(
    "Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv"
)

# ====================================
# DATASET
# ====================================
st.subheader("📂 Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric("Jumlah Data", len(df))

with col2:
    st.metric("Jumlah Kolom", len(df.columns))

st.dataframe(df.head())

# ====================================
# PREPROCESSING
# ====================================
data = df.copy()

data.drop(
    columns=[
        "transaction_id",
        "user_id",
        "addiction_level"
    ],
    errors="ignore",
    inplace=True
)

gender_encoder = None

for col in data.select_dtypes(include="object").columns:

    le = LabelEncoder()

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

    if col == "gender":
        gender_encoder = le

# ====================================
# TARGET
# ====================================
target = "addicted_label"

X = data.drop(columns=[target])
y = data[target]

# ====================================
# SPLIT DATA
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ====================================
# MODEL
# ====================================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ====================================
# EVALUASI
# ====================================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(
    y_train,
    train_pred
)

test_acc = accuracy_score(
    y_test,
    test_pred
)

# ====================================
# HASIL AKURASI
# ====================================
st.subheader("📊 Hasil Akurasi")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Akurasi Training",
        f"{train_acc*100:.2f}%"
    )

with c2:
    st.metric(
        "Akurasi Testing",
        f"{test_acc*100:.2f}%"
    )

# ====================================
# FEATURE IMPORTANCE
# ====================================
st.subheader("📌 Feature Importance")

importance = pd.DataFrame({
    "Fitur": X.columns,
    "Nilai": model.feature_importances_
})

importance = importance.sort_values(
    by="Nilai",
    ascending=False
)

st.dataframe(
    importance,
    use_container_width=True
)

# ====================================
# PREDIKSI USER
# ====================================
st.subheader("🔍 Prediksi Kecanduan Smartphone")

with st.form("form_prediksi"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Umur",
            min_value=10,
            max_value=100,
            value=20
        )

        gender = st.selectbox(
            "Jenis Kelamin",
            gender_encoder.classes_
        )

        daily_screen_time_hours = st.number_input(
            "Daily Screen Time Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0
        )

        social_media_hours = st.number_input(
            "Social Media Hours",
            min_value=0.0,
            max_value=24.0,
            value=2.0
        )

        gaming_hours = st.number_input(
            "Gaming Hours",
            min_value=0.0,
            max_value=24.0,
            value=1.0
        )

        work_study_hours = st.number_input(
            "Work Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0
        )

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=8.0
        )

        notifications_per_day = st.number_input(
            "Notifications Per Day",
            min_value=0,
            max_value=1000,
            value=100
        )

        app_opens_per_day = st.number_input(
            "App Opens Per Day",
            min_value=0,
            max_value=1000,
            value=50
        )

        weekend_screen_time = st.number_input(
            "Weekend Screen Time",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

        stress_level = st.number_input(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=5
        )

        academic_work_impact = st.number_input(
            "Academic Work Impact",
            min_value=1,
            max_value=10,
            value=5
        )

    submit = st.form_submit_button("Prediksi")

if submit:

    gender_value = gender_encoder.transform([gender])[0]

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender_value,
        "daily_screen_time_hours": daily_screen_time_hours,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact
    }])

    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    hasil = model.predict(input_data)[0]
    probabilitas = model.predict_proba(input_data)[0]

    prob_kecanduan = probabilitas[1] * 100

    st.markdown("""
    <style>
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .card-title {
        font-size: 20px;
        font-weight: bold;
    }

    .card-value {
        font-size: 35px;
        font-weight: bold;
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

    if prob_kecanduan >= 50:

        st.error(
            f"⚠️ Hasil Prediksi : Kecanduan Smartphone ({prob_kecanduan:.2f}%)"
        )

        st.info("""
💡 Rekomendasi:

• Kurangi penggunaan smartphone secara bertahap.
• Batasi penggunaan media sosial maksimal 2–3 jam per hari.
• Aktifkan fitur Screen Time atau Digital Wellbeing.
• Perbanyak aktivitas fisik dan interaksi sosial secara langsung.
• Hindari penggunaan smartphone sebelum tidur.
• Atur jadwal belajar atau bekerja tanpa gangguan notifikasi.
""")

    else:

        st.success(
            f"✅ Hasil Prediksi : Tidak Kecanduan Smartphone ({prob_kecanduan:.2f}%)"
        )

        st.info("""
💡 Rekomendasi:

• Pertahankan kebiasaan penggunaan smartphone yang sehat.
• Tetap menjaga keseimbangan antara aktivitas online dan offline.
• Gunakan smartphone sesuai kebutuhan.
• Pertahankan pola tidur dan aktivitas harian yang baik.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    🟢 Normal
                </div>
                <div class="card-value">
                    {probabilitas[0]*100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    🔴 Kecanduan
                </div>
                <div class="card-value">
                    {probabilitas[1]*100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # ====================================
    # TINGKAT RISIKO
    # ====================================
    st.subheader("📈 Tingkat Risiko")

    if prob_kecanduan >= 80:
        st.error("🔴 Risiko Sangat Tinggi")

    elif prob_kecanduan >= 60:
        st.warning("🟠 Risiko Tinggi")

    elif prob_kecanduan >= 40:
        st.info("🟡 Risiko Sedang")

    else:
        st.success("🟢 Risiko Rendah")

    # ====================================
    # PROGRESS BAR
    # ====================================
    st.subheader("📊 Persentase Risiko")

    st.progress(int(prob_kecanduan))

    st.write(
        f"Tingkat Risiko Kecanduan Smartphone : {prob_kecanduan:.2f}%"
    )

   # ====================================
# GRAFIK HASIL PREDIKSI
# ====================================


        # ====================================
    # GRAFIK HASIL PREDIKSI
    # ====================================
    st.subheader("📊 Grafik Hasil Prediksi")

    chart_data = pd.DataFrame({
        "Kategori": ["Normal", "Kecanduan"],
        "Persentase": [
            round(probabilitas[0] * 100, 2),
            round(probabilitas[1] * 100, 2)
        ]
    })

    fig = px.bar(
        chart_data,
        x="Kategori",
        y="Persentase",
        text="Persentase",
        color="Kategori",
        color_discrete_map={
            "Normal": "#22c55e",
            "Kecanduan": "#ef4444"
        }
    )

    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="Kategori",
        yaxis_title="Persentase (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    # ====================================
    # FAKTOR RISIKO
    # ====================================
    st.subheader("📌 Faktor Penyebab Utama")

    faktor = []

    if daily_screen_time_hours >= 8:
        faktor.append("Screen Time Harian Tinggi")

    if social_media_hours >= 4:
        faktor.append("Penggunaan Media Sosial Berlebihan")

    if gaming_hours >= 3:
        faktor.append("Durasi Bermain Game Tinggi")

    if sleep_hours <= 6:
        faktor.append("Kurang Tidur")

    if notifications_per_day >= 200:
        faktor.append("Notifikasi Smartphone Sangat Banyak")

    if app_opens_per_day >= 100:
        faktor.append("Frekuensi Membuka Aplikasi Sangat Tinggi")

    if stress_level >= 7:
        faktor.append("Tingkat Stress Tinggi")

    if academic_work_impact >= 7:
        faktor.append("Aktivitas Akademik/Tugas Terganggu")

    if len(faktor) > 0:

        for item in faktor:
            st.write("•", item)

    else:

        st.success(
            "Tidak ditemukan faktor risiko yang signifikan."
        )

    # ====================================
    # DOWNLOAD HASIL
    # ====================================
    st.subheader("📥 Download Hasil Prediksi")

    hasil_download = f"""
    HASIL PREDIKSI KECANDUAN SMARTPHONE

    Probabilitas Tidak Kecanduan :
    {probabilitas[0]*100:.2f}%

    Probabilitas Kecanduan :
    {prob_kecanduan:.2f}%

    Hasil :
    {'Kecanduan Smartphone' if prob_kecanduan >= 50 else 'Tidak Kecanduan Smartphone'}
    """

    st.download_button(
        label="📄 Download Hasil",
        data=hasil_download,
        file_name="hasil_prediksi.txt",
        mime="text/plain"
    )

    # ====================================
    # RIWAYAT PREDIKSI
    # ====================================
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Umur": age,
        "Screen Time": daily_screen_time_hours,
        "Media Sosial": social_media_hours,
        "Risiko (%)": round(prob_kecanduan, 2)
    })

    st.subheader("📋 Riwayat Prediksi")

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True
    )

    # ====================================
    # INTERPRETASI HASIL
    # ====================================
    st.subheader("📝 Interpretasi Hasil")

    if prob_kecanduan >= 50:

        st.warning(
            "Model Decision Tree mendeteksi adanya kecenderungan kecanduan smartphone berdasarkan pola penggunaan yang dimasukkan."
        )

    else:

        st.success(
            "Model Decision Tree mendeteksi pola penggunaan smartphone masih dalam batas normal dan terkendali."
        )

        st.subheader("📋 Analisis Input")

        skor = 0

        if daily_screen_time_hours >= 8:
            skor += 1

        if social_media_hours >= 4:
            skor += 1

        if gaming_hours >= 3:
            skor += 1

        if sleep_hours <= 6:
            skor += 1

        if notifications_per_day >= 200:
            skor += 1

        if app_opens_per_day >= 100:
            skor += 1

        if stress_level >= 7:
            skor += 1

        if academic_work_impact >= 7:
            skor += 1

        if skor >= 4:
            st.warning(
                "Berdasarkan data yang dimasukkan, pola penggunaan smartphone tergolong cukup tinggi dan berpotensi menyebabkan kecanduan."
            )
        else:
            st.success(
                "Berdasarkan data yang dimasukkan, pola penggunaan smartphone masih dalam kategori normal dan terkendali."
            )
    
    # ====================================
    # KESIMPULAN
    # ====================================
        st.subheader("📝 Kesimpulan")

        if test_acc >= 0.90:
            kategori = "Sangat Baik"

        elif test_acc >= 0.80:
            kategori = "Baik"

        else:
            kategori = "Cukup"

        st.info(
            f"""
        Dataset sebanyak {len(df)} data berhasil
        diklasifikasikan menggunakan algoritma
        Decision Tree.

        Akurasi Training : {train_acc*100:.2f}%

        Akurasi Testing : {test_acc*100:.2f}%

        Kategori Model : {kategori}
        """
        )
