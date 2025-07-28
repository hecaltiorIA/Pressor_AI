
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predicción de Hipotensión", layout="centered")
st.title("💉 Predicción de uso de vasopresores")
st.write("Llena los datos del paciente y presiona el botón para calcular el riesgo.")

@st.cache_data
def cargar_y_entrenar_modelo():
    url = "https://raw.githubusercontent.com/hecaltiorIA/modelo-hipotension/main/X_train_2025.csv"
    df = pd.read_csv(url)
    df['MAP_low'] = (df['MAP_lowest'] < 65).astype(int)
    df.drop(columns=['recordid', 'MAP_lowest'], inplace=True)

    selected_features = [
        'DiasABP_first', 'SysABP_first', 'SAPS-I', 'GCS_lowest', 'PaO2_first', 'CSRU',
        'HR_first', 'GCS_first', 'GCS_last', 'Temp_first',
        'FiO2_first', 'Creatinine_first', 'Lactate_first',
        'K_first', 'Na_first', 'WBC_first'
    ]

    X = df[selected_features]
    y = df['MAP_low']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, imputer, scaler, selected_features

modelo, imputador, escalador, campos = cargar_y_entrenar_modelo()

    etiquetas = {
        "DiasABP_first": "Presión diastólica inicial",
        "SysABP_first": "Presión sistólica inicial",
        "SAPS-I": "Escala SAPS-I",
        "GCS_lowest": "Glasgow más bajo",
        "PaO2_first": "PaO2 inicial",
        "CSRU": "Unidad de recuperación cardíaca (0 = No, 1 = Sí)",
        "HR_first": "Frecuencia cardíaca inicial",
        "GCS_first": "Glasgow al ingreso",
        "GCS_last": "Glasgow al egreso",
        "Temp_first": "Temperatura inicial",
        "FiO2_first": "FiO2 inicial",
        "Creatinine_first": "Creatinina inicial",
        "Lactate_first": "Lactato inicial",
        "K_first": "Potasio",
        "Na_first": "Sodio",
        "WBC_first": "Leucocitos",
    }

valores = []
for campo in campos:
    if campo == "CSRU":
        valor = st.number_input("Unidad de recuperación cardíaca (0 = No, 1 = Sí)", key="CSRU", step=1, min_value=0, max_value=1)
    else:
        valor = st.number_input(etiquetas.get(campo, campo), step=0.1, format="%.2f")
    valores.append(valor)

if st.button("Calcular riesgo"):
    entrada = np.array(valores).reshape(1, -1)
    entrada_imputada = imputador.transform(entrada)
    entrada_escalada = escalador.transform(entrada_imputada)
    prob = modelo.predict_proba(entrada_escalada)[0][1]
    df_usuario = pd.DataFrame([entrada_convertida])
    riesgo = modelo.predict(df_usuario)[0]
    if riesgo == 1:
        st.error("⚠️ Riesgo alto de requerir vasopresores (TAM < 65 mmHg)")
    else:
        st.success("✅ Bajo riesgo de requerir vasopresores")
