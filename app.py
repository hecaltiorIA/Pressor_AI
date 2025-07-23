
import streamlit as st
import joblib
import numpy as np

# Cargar modelo, escalador e imputador
modelo = joblib.load("modelo_random_forest.pkl")
escalador = joblib.load("escalador.pkl")
imputador = joblib.load("imputador.pkl")

st.set_page_config(page_title="Predicción de Hipotensión", layout="centered")
st.title("🩺 Predicción de Hipotensión (TAM < 65 mmHg)")
st.write("Llena los datos del paciente y presiona el botón para calcular el riesgo.")

campos = [
    "DiasABP_first", "SysABP_first", "SAPS-I", "GCS_lowest", "PaO2_first", "CSRU",
    "HR_first", "GCS_first", "GCS_last", "Temp_first",
    "FiO2_first", "Creatinine_first", "Lactate_first",
    "K_first", "Na_first", "WBC_first"
]

valores = []
for campo in campos:
    if campo == "CSRU":
        valor = st.selectbox("¿El paciente está en CSRU (unidad de recuperación cardiaca)?", [0, 1])
    else:
        valor = st.number_input(campo, step=0.1, format="%.2f")
    valores.append(valor)

if st.button("Calcular riesgo"):
    entrada = np.array(valores).reshape(1, -1)
    entrada_imputada = imputador.transform(entrada)
    entrada_escalada = escalador.transform(entrada_imputada)
    prob = modelo.predict_proba(entrada_escalada)[0][1]
    riesgo = modelo.predict(entrada_escalada)[0]

    st.subheader("🧠 Resultado del modelo:")
    if riesgo == 1:
        st.error(f"⚠️ Alto riesgo de hipotensión. Probabilidad: {prob:.2f}")
    else:
        st.success(f"✅ Bajo riesgo de hipotensión. Probabilidad: {prob:.2f}")
