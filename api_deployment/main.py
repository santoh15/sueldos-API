from fastapi import FastAPI  
from dotenv import load_dotenv 
import psycopg2
import os
from api_deployment.interface import router_interface
from api_deployment.predict import router_predict

load_dotenv()

app = FastAPI(title="API IT Salary Predictor", description="API for predicting IT salaries", version="1.0.0")
DATABASE_URL = os.getenv("DATABASE_URL")

def setup_db():
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historial_predicciones (
                id SERIAL PRIMARY KEY,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                seniority VARCHAR(50),
                anos_en_el_puesto_actual REAL,
                cuantas_personas_tenes_a_cargo INTEGER,
                recibis_algun_tipo_de_bono VARCHAR(255),
                tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre VARCHAR(255),
                sueldo_dolarizado BOOLEAN,
                dedicacion VARCHAR(50),
                anos_de_experiencia REAL,
                tengo_edad INTEGER,
                trabajo_de VARCHAR(255),
                lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual TEXT,
                sueldo_real_percibido REAL,
                estimated_salary_ars REAL
            )
        """)
        conn.commit()
        print("Database setup completed successfully.")
    except Exception as e:
        print(f"Error setting up database: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

setup_db()
app.include_router(router_interface)
app.include_router(router_predict)
