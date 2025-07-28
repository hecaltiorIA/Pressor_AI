
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
    url = "https://raw.githubusercontent.com/hecaltiorIA/Pressor_AI/main/X_train_2025.csv"
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
    "GCS_last": "Último Glasgow",
    "Temp_first": "Temperatura inicial",
    "FiO2_first": "FiO2 inicial",
    "Creatinine_first": "Creatinina inicial",
    "Lactate_first": "Lactato inicial",
    "K_first": "Potasio",
    "Na_first": "Sodio",
    "WBC_first": "Leucocitos",
}

for campo in campos:
    if campo == "CSRU":
        st.number_input(etiquetas.get(campo, campo), key=campo, step=1, min_value=0, max_value=1)
    else:
        st.number_input(etiquetas.get(campo, campo), key=campo, step=0.1, format="%.2f")

if st.button("Calcular riesgo"):
    valores_usuario = {campo: st.session_state.get(campo, 0) for campo in campos}

    if any(valor is None or (isinstance(valor, (int, float)) and valor == 0 and campo != "CSRU") for campo, valor in valores_usuario.items()):
        st.warning("⚠️ Por favor, llena todos los campos del formulario antes de calcular el riesgo.")
    else:
        df_usuario = pd.DataFrame([valores_usuario])
        df_usuario_imputado = imputador.transform(df_usuario)
        df_usuario_escalado = escalador.transform(df_usuario_imputado)
        riesgo = modelo.predict(df_usuario_escalado)[0]
        probabilidad = modelo.predict_proba(df_usuario_escalado)[0][1]

    # Mostrar resultado solo como porcentaje con color en fondo negro
    color = "red" if probabilidad > 0.85 else "orange" if probabilidad > 0.65 else "yellow" if probabilidad > 0.45 else "green"
    st.markdown(f"""
        <h2 style='color:{color}; background-color:black; text-align:center;'>
        Probabilidad: {probabilidad:.2%}
        </h2>""", unsafe_allow_html=True)
