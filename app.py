import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================
# CONFIG
# =====================
st.set_page_config(
    page_title="Data Mining Smartphone",
    layout="wide"
)

st.title("📱 Analisis Kecanduan Smartphone Menggunakan Decision Tree")

# =====================
# LOAD DATASET
# =====================
try:
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv")

    st.success("Dataset berhasil dimuat")

    # =====================
    # PREVIEW DATASET
    # =====================
    st.header("1. Preview Dataset")
    st.dataframe(df)

    # =====================
    # INFORMASI DATASET
    # =====================
    st.header("2. Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Baris", df.shape[0])

    with col2:
        st.metric("Jumlah Kolom", df.shape[1])

    # =====================
    # MISSING VALUE
    # =====================
    st.header("3. Missing Value")

    missing = pd.DataFrame(
        df.isnull().sum(),
        columns=["Jumlah Missing"]
    )

    st.dataframe(missing)

    # =====================
    # VISUALISASI DATA
    # =====================
    st.header("4. Visualisasi Data")

    if "addicted_label" in df.columns:

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.countplot(
            x="addicted_label",
            data=df,
            ax=ax
        )

        plt.title("Distribusi Label Kecanduan Smartphone")
        st.pyplot(fig)

    # =====================
    # PREPROCESSING
    # =====================
    st.header("5. Preprocessing Data")

    df_model = df.copy()

    categorical_columns = [
        "gender",
        "stress_level",
        "academic_work_impact",
        "addiction_level",
        "addicted_label"
    ]

    for col in categorical_columns:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])

    st.success("Preprocessing berhasil dilakukan")

    # =====================
    # FEATURE & TARGET
    # =====================
    drop_columns = []

    if "transaction_id" in df_model.columns:
        drop_columns.append("transaction_id")

    if "user_id" in df_model.columns:
        drop_columns.append("user_id")

    if "addicted_label" in df_model.columns:
        drop_columns.append("addicted_label")

    X = df_model.drop(columns=drop_columns)
    y = df_model["addicted_label"]

    # =====================
    # SPLIT DATA
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================
    # DECISION TREE
    # =====================
    st.header("6. Training Decision Tree")

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    model.fit(X_train, y_train)

    st.success("Model berhasil dilatih")

    # =====================
    # VISUALISASI TREE
    # =====================
    st.header("7. Visualisasi Decision Tree")

    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))

    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Tidak Addicted", "Addicted"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )

    st.pyplot(fig_tree)

    # =====================
    # RULES
    # =====================
    st.header("8. Aturan Decision Tree")

    rules = export_text(
        model,
        feature_names=list(X.columns)
    )

    st.text(rules)

    # =====================
    # PREDIKSI
    # =====================
    y_pred = model.predict(X_test)

    # =====================
    # AKURASI
    # =====================
    st.header("9. Akurasi Model")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Accuracy",
        f"{accuracy * 95,20:.2f}%"
    )

    # =====================
    # CONFUSION MATRIX
    # =====================
    st.header("10. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm
    )

    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")

    st.pyplot(fig_cm)

    # =====================
    # CLASSIFICATION REPORT
    # =====================
    st.header("11. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

except FileNotFoundError:
    st.error(
        "File 'smartphone_addiction.csv' tidak ditemukan. "
        "Pastikan file CSV berada pada folder yang sama dengan app.py."
    )
