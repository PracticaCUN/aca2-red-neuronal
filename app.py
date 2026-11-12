import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Cargar modelo y artefactos
model = tf.keras.models.load_model("student_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

# Diccionario opcional para traducir los nombres más comunes al español
traducciones_columnas = {
    "Marital status": "Estado civil",
    "Application mode": "Modo de aplicación",
    "Application order": "Orden de aplicación",
    "Course": "Programa/Curso",
    "Daytime/evening attendance": "Asistencia (Diurna/Nocturna)",
    "Previous qualification": "Titulación previa",
    "Previous qualification (grade)": "Nota de titulación previa",
    "Nacionality": "Nacionalidad",
    "Mother's qualification": "Titulación de la madre",
    "Father's qualification": "Titulación del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Nota de admisión",
    "Displaced": "Desplazado",
    "Educational special needs": "Necesidades educativas especiales",
    "Debtor": "Deudor",
    "Tuition fees up to date": "Matrícula al día",
    "Gender": "Género",
    "Scholarship holder": "Becario",
    "Age at enrollment": "Edad al matricularse",
    "International": "Estudiante internacional",
    "Curricular units 1st sem (credited)": "Unidades 1er sem (crediditadas)",
    "Curricular units 1st sem (enrolled)": "Unidades 1er sem (matriculadas)",
    "Curricular units 1st sem (evaluations)": "Unidades 1er sem (evaluaciones)"
}

st.title("🎓 Predicción de Deserción y Éxito Académico")
st.write("Esta aplicación utiliza una **Red Neuronal Artificial (MLP)** entrenada con el dataset de la UCI para predecir el estado académico de un estudiante.")

st.sidebar.header("Parámetros del Estudiante")
st.sidebar.write("Ingresa los datos requeridos:")

input_data = {}
for col in model_columns:
    # Mostrar el nombre traducido si existe, de lo contrario mostrar el original
    label_mostrar = traducciones_columnas.get(col, col)
    input_data[col] = st.sidebar.number_input(label_mostrar, value=0.0, step=1.0, format="%.0f")

if st.button("Predecir Estado Académico"):
    # Convertir a DataFrame y escalar
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    # Realizar predicción
    prediction = model.predict(input_scaled)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = float(np.max(prediction) * 100)
    
    # Mostrar resultados mejorados
    st.success(f"### Resultado Predicho: {predicted_class}")
    st.info(f"**Confianza del modelo:** {confidence:.2f}%")
    
    st.subheader("📊 Probabilidades detalladas por categoría:")
    
    # Crear un gráfico de barras más limpio con pandas/streamlit nativo
    prob_df = pd.DataFrame({
        'Categoría': label_encoder.classes_,
        'Probabilidad': prediction[0]
    }).set_index('Categoría')
    
    st.bar_chart(prob_df)
