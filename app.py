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

st.set_page_config(
page_title="Data Mining Smartphone",
layout="wide"
)

st.title("📱 Analisis Kecanduan Smartphone Menggunakan Decision Tree")

# =====================

# LOAD DATASET

# =====================

df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500.csv")
st.success("Dataset berhasil dimuat")

st.error(f"Gagal membaca dataset: {e}")
st.stop()

# =====================

# PREVIEW DATASET

# =====================

st.header("1. Preview Dataset")
st.dataframe(df.head())

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

```
sns.countplot(
    x="addicted_label",
    data=df,
    ax=ax
)

st.pyplot(fig)
```

# =====================

# PREPROCESSING

# =====================

st.header("5. Preprocessing")

le = LabelEncoder()

categorical_columns = [
"gender",
"stress_level",
"academic_work_impact",
"addiction_level",
"addicted_label"
]

for col in categorical_columns:
if col in df.columns:
df[col] = le.fit_transform(df[col])

st.success("Preprocessing berhasil")
