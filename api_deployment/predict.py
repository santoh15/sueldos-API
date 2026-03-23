import sys
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import psycopg2
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from src.encoding import encode_data, encode_language 

router_predict = APIRouter()

model_path = os.path.join(CURRENT_DIR, 'model_xgb.pkl')
columns_path = os.path.join(CURRENT_DIR, 'columns.pkl')

model = joblib.load(model_path)
columns_train = joblib.load(columns_path)

DATABASE_URL = os.getenv("DATABASE_URL")
class userdata(BaseModel):
    dedicacion: str
    recibis_algun_tipo_de_bono: str
    tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre: int
    trabajo_de: str
    anos_de_experiencia: float
    antiguedad_en_la_empresa_actual: float
    anos_en_el_puesto_actual: float
    cuantas_personas_tenes_a_cargo: int
    lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual: str
    tengo_edad: int
    sueldo_dolarizado: int
    seniority: str
    
    sueldo_real_percibido: Optional[float] = None


@router_predict.post("/predict")
def predict_salary(datos: userdata):
    df_in = pd.DataFrame([datos.dict()])

    df_in = encode_language(df_in)
    df_in = encode_data(df_in)
    df_model = df_in.reindex(columns=columns_train, fill_value=0)

    prediction_log = model.predict(df_model)
    estimated_salary_ars = float(round(np.expm1(prediction_log[0]), 2))
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        consult_sql = """
            INSERT INTO historial_predicciones (
                dedicacion, recibis_algun_tipo_de_bono, tuviste_actualizaciones_durante_ultimo_semestre,
                trabajo_de, anos_de_experiencia, antiguedad_en_la_empresa_actual, anos_en_el_puesto_actual,
                cuantas_personas_tenes_a_cargo, lenguajes_de_programacion, tengo_edad,
                sueldo_dolarizado, seniority, sueldo_real_percibido, estimated_salary_ars
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            datos.dedicacion, datos.recibis_algun_tipo_de_bono, datos.tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre,
            datos.trabajo_de, datos.anos_de_experiencia, datos.antiguedad_en_la_empresa_actual, datos.anos_en_el_puesto_actual,
            datos.cuantas_personas_tenes_a_cargo, datos.lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual, datos.tengo_edad,
            datos.sueldo_dolarizado, datos.seniority, datos.sueldo_real_percibido, estimated_salary_ars
        )
        
        cursor.execute(consult_sql, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error saving to SQL: {e}")
    return {"sueldo_estimado_ars": estimated_salary_ars}