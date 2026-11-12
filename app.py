
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf

@st.cache_resource
def cargar():
    return (
        tf.keras.models.load_model("student_model.h5"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoder.pkl"),
        joblib.load("model_columns.pkl")
    )

model, scaler, label_encoder, model_columns = cargar()

stats = pd.read_json("feature_stats.json")

st.title("🎓 Predicción de Deserción y Éxito Académico")

st.sidebar.header("Parámetros del Estudiante")

input_data = {}

for col in model_columns:

    mn = float(stats.loc[col, 'min'])
    mx = float(stats.loc[col, 'max'])
    med = float(stats.loc[col, '50%'])

    input_data[col] = st.sidebar.number_input(
        col,
        min_value=mn,
        max_value=mx,
        value=med
    )

if st.button("Predecir"):

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)

    idx = np.argmax(pred)

    resultado = label_encoder.inverse_transform([idx])[0]

    st.success(f"Resultado: {resultado}")

    st.bar_chart(
        pd.DataFrame(
            pred[0],
            index=label_encoder.classes_,
            columns=["Probabilidad"]
        )
    )
