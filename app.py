
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Deserción y Éxito Académico", page_icon="🎓", layout="wide")

@st.cache_resource
def cargar():
    return (tf.keras.models.load_model("student_model.keras"),
            joblib.load("scaler.pkl"),
            joblib.load("label_encoder.pkl"),
            joblib.load("model_columns.pkl"))

model, scaler, label_encoder, model_columns = cargar()
stats = pd.read_json("feature_stats.json")
stats.index = stats.index.str.strip()

TR = {
    "Marital status": "Estado civil (código)", "Application mode": "Modo de aplicación (código)",
    "Application order": "Orden de aplicación", "Course": "Programa/Curso (código)",
    "Daytime/evening attendance": "Asistencia (1=diurna, 0=nocturna)",
    "Previous qualification": "Titulación previa (código)",
    "Previous qualification (grade)": "Nota de titulación previa",
    "Nacionality": "Nacionalidad (código)",
    "Mother's qualification": "Titulación de la madre (código)",
    "Father's qualification": "Titulación del padre (código)",
    "Mother's occupation": "Ocupación de la madre (código)",
    "Father's occupation": "Ocupación del padre (código)",
    "Admission grade": "Nota de admisión", "Displaced": "Desplazado (1=sí)",
    "Educational special needs": "Necesidades educativas especiales (1=sí)",
    "Debtor": "Deudor (1=sí)", "Tuition fees up to date": "Matrícula al día (1=sí)",
    "Gender": "Género (1=masculino, 0=femenino)", "Scholarship holder": "Becario (1=sí)",
    "Age at enrollment": "Edad al matricularse", "International": "Estudiante internacional (1=sí)",
    "Curricular units 1st sem (credited)": "1er sem: unidades acreditadas",
    "Curricular units 1st sem (enrolled)": "1er sem: unidades matriculadas",
    "Curricular units 1st sem (evaluations)": "1er sem: evaluaciones",
    "Curricular units 1st sem (approved)": "1er sem: unidades aprobadas",
    "Curricular units 1st sem (grade)": "1er sem: nota promedio",
    "Curricular units 1st sem (without evaluations)": "1er sem: unidades sin evaluación",
    "Curricular units 2nd sem (credited)": "2.º sem: unidades acreditadas",
    "Curricular units 2nd sem (enrolled)": "2.º sem: unidades matriculadas",
    "Curricular units 2nd sem (evaluations)": "2.º sem: evaluaciones",
    "Curricular units 2nd sem (approved)": "2.º sem: unidades aprobadas",
    "Curricular units 2nd sem (grade)": "2.º sem: nota promedio",
    "Curricular units 2nd sem (without evaluations)": "2.º sem: unidades sin evaluación",
    "Unemployment rate": "Tasa de desempleo (%)", "Inflation rate": "Tasa de inflación (%)",
    "GDP": "PIB (variación)",
}

st.title("🎓 Predicción de Deserción y Éxito Académico")
st.write("Perceptrón Multicapa (MLP) entrenado con el dataset UCI *Predict Students' Dropout and Academic Success*. "
         "Los campos inician con la **mediana** del dataset; ajústalos para simular un estudiante.")

faltantes = [c for c in model_columns if c not in stats.index]
if faltantes:
    st.error(f"feature_stats.json no coincide con el modelo. Faltan: {faltantes}")
    st.stop()

st.sidebar.header("Parámetros del Estudiante")
input_data = {}
for col in model_columns:
    mn, med, mx = (float(stats.loc[col, k]) for k in ("min", "50%", "max"))
    label = TR.get(col, col)
    if mn.is_integer() and mx.is_integer() and med.is_integer():
        input_data[col] = st.sidebar.number_input(label, min_value=int(mn), max_value=int(mx),
                                                  value=int(med), step=1)
    else:
        input_data[col] = st.sidebar.number_input(label, min_value=mn, max_value=mx,
                                                  value=med, step=0.1, format="%.2f")

if st.button("Predecir Estado Académico"):
    input_df = pd.DataFrame([input_data])[model_columns]
    pred = model.predict(scaler.transform(input_df), verbose=0)
    idx = int(np.argmax(pred))
    st.success(f"### Resultado predicho: {label_encoder.inverse_transform([idx])[0]}")
    st.info(f"**Confianza del modelo:** {pred[0][idx]*100:.2f}%")
    st.subheader("📊 Probabilidades por categoría")
    st.bar_chart(pd.DataFrame({"Probabilidad": pred[0]}, index=label_encoder.classes_))

st.caption("Herramienta académica de apoyo; no sustituye la valoración institucional.")
