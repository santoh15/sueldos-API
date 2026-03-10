from fastapi import FastAPI
from dotenv import load_dotenv
import psycopg2
import os

# Importamos nuestros archivos separados
from interfaz import router_interfaz
from predecir import router_predecir

load_dotenv()

app = FastAPI(title="API Predictor de Sueldos IT")

# --- CONFIGURACIÓN DE BASE DE DATOS ---
DATABASE_URL = os.getenv("DATABASE_URL")

def inicializar_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historial_predicciones (
                id SERIAL PRIMARY KEY,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                donde_estas_trabajando VARCHAR(255),
                dedicacion VARCHAR(50),
                modalidad_de_trabajo VARCHAR(50),
                genero VARCHAR(50),
                seniority VARCHAR(50),
                recibis_algun_tipo_de_bono INTEGER,
                tuviste_actualizaciones_durante_ultimo_semestre INTEGER,
                estas_buscando_trabajo INTEGER,
                cuantas_personas_tenes_a_cargo INTEGER,
                sueldo_dolarizado INTEGER,
                anos_de_experiencia REAL,
                antiguedad_en_la_empresa_actual REAL,
                anos_en_el_puesto_actual REAL,
                tengo_edad INTEGER,
                contas_con_beneficios_adicionales TEXT,
                trabajo_de VARCHAR(255),
                lenguajes_de_programacion TEXT,
                sueldo_real_percibido REAL,
                sueldo_estimado_modelo REAL
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Base de datos conectada y verificada")
    except Exception as e:
        print(f"Error conectando a la BD: {e}")

inicializar_db()

# --- CONECTAR LAS RUTAS ---
app.include_router(router_interfaz)
app.include_router(router_predecir)
