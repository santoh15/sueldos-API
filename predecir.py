from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import psycopg2
import os

router_predecir = APIRouter()


modelo = joblib.load('modelo_xgb.pkl')
columnas_entrenamiento = joblib.load('columnas.pkl')
DATABASE_URL = os.getenv("DATABASE_URL")


class DatosUsuario(BaseModel):
    donde_estas_trabajando: str
    dedicacion: str
    modalidad_de_trabajo: str
    genero: str
    seniority: str
    recibis_algun_tipo_de_bono: int 
    tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre: int
    estas_buscando_trabajo: int
    cuantas_personas_tenes_a_cargo: int 
    sueldo_dolarizado: int  
    anos_de_experiencia: float
    antiguedad_en_la_empresa_actual: float
    anos_en_el_puesto_actual: float
    tengo_edad: int
    contas_con_beneficios_adicionales: str
    trabajo_de: str
    lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual: str
    sueldo_real_percibido: Optional[float] = None


@router_predecir.post("/predecir")
def predecir_sueldo(datos: DatosUsuario):
    df_entrada = pd.DataFrame([datos.dict()])
    columnas_texto = ['donde_estas_trabajando', 'dedicacion', 'modalidad_de_trabajo', 'genero', 'seniority', 'contas_con_beneficios_adicionales', 'trabajo_de', 'lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual']
    
    for col in columnas_texto:
        df_entrada[col] = df_entrada[col].str.lower().str.replace(' ', '_')
    
    df_dummies = pd.get_dummies(df_entrada, dtype=int)
    df_modelo = df_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    prediccion_log = modelo.predict(df_modelo)
    sueldo_estimado_ars = float(round(np.expm1(prediccion_log[0]), 2))
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        consulta_sql = """
            INSERT INTO historial_predicciones (
                donde_estas_trabajando, dedicacion, modalidad_de_trabajo, genero, seniority,
                recibis_algun_tipo_de_bono, tuviste_actualizaciones_durante_ultimo_semestre,
                estas_buscando_trabajo, cuantas_personas_tenes_a_cargo, sueldo_dolarizado,
                anos_de_experiencia, antiguedad_en_la_empresa_actual, anos_en_el_puesto_actual,
                tengo_edad, contas_con_beneficios_adicionales, trabajo_de,
                lenguajes_de_programacion, sueldo_real_percibido, sueldo_estimado_modelo
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        valores = (
            datos.donde_estas_trabajando, datos.dedicacion, datos.modalidad_de_trabajo, datos.genero, datos.seniority,
            datos.recibis_algun_tipo_de_bono, datos.tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre,
            datos.estas_buscando_trabajo, datos.cuantas_personas_tenes_a_cargo, datos.sueldo_dolarizado,
            datos.anos_de_experiencia, datos.antiguedad_en_la_empresa_actual, datos.anos_en_el_puesto_actual,
            datos.tengo_edad, datos.contas_con_beneficios_adicionales, datos.trabajo_de,
            datos.lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual,
            datos.sueldo_real_percibido, sueldo_estimado_ars
        )
        
        cursor.execute(consulta_sql, valores)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error al guardar en SQL: {e}")
    return {"sueldo_estimado_ars": sueldo_estimado_ars}