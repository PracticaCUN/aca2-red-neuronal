import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Predicción de Éxito Académico", page_icon="🎓", layout="centered")

# Cargar el modelo y los artefactos guardados
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('student_model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, label_encoder, model_columns

try:
    model, scaler, label_encoder, model_columns = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")
    st.stop()

st.title("🎓 Predicción de Deserción y Éxito Académico")
st.write("Esta aplicación utiliza una **Red Neuronal Artificial (MLP)** entrenada con el dataset de la UCI para predecir el estado académico de un estudiante.")

st.sidebar.header("Parámetros del Estudiante")
st.sidebar.write("Ingresa los datos requeridos para realizar la predicción:")

# Como el dataset tiene muchas columnas, crearemos una entrada interactiva genérica 
# o basada en los valores por defecto de las columnas que el modelo espera.
input_data = {}
for col in model_columns:
    # Usamos step=1 y formato entero para evitar decimales innecesarios
    input_data[col] = st.sidebar.number_input(f"{col}", value=0.0, step=1.0, format="%.0f")

# Botón de predicción
if st.button("Predecir Estado Académico", type="primary"):
    # Convertir los datos ingresados a DataFrame asegurando el orden exacto de las columnas
    input_df = pd.DataFrame([input_data])
    
    # Escalar los datos igual que en el entrenamiento
    input_scaled = scaler.transform(input_df)
    
    # Realizar la predicción con la red neuronal
    prediction_probs = model.predict(input_scaled)
    predicted_class_idx = np.argmax(prediction_probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = np.max(prediction_probs) * 100
    
    # Mostrar resultados en pantalla
    st.success(f"### Resultado Predicho: **{predicted_label}**")
    st.info(f"Confianza del modelo: **{confidence:.2f}%**")
    
    # Mostrar desglose de probabilidades por clase
    st.write("#### Probabilidades detalladas:")
    prob_df = pd.DataFrame(prediction_probs, columns=label_encoder.classes_)
    st.bar_chart(prob_df.T)
