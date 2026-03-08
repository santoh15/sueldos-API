# API de Predicción Salarial IT (End-to-End ML)

Un servicio web End-to-End que predice sueldos del sector tecnológico en Argentina mediante Machine Learning. El proyecto abarca desde el entrenamiento del modelo hasta su despliegue en la nube y el almacenamiento de inferencias en tiempo real.

🔗 **[Probá la API en vivo acá](https://api-sueldos-873271753459.us-central1.run.app/)**

## Arquitectura del Sistema
1. **Frontend / Cliente:** Interfaz HTML/JS servida directamente desde la API para ingresar los 14 parámetros del usuario.
2. **Procesamiento (FastAPI):** Recibe los datos, limpia las cadenas de texto y aplica One-Hot Encoding dinámico utilizando Pandas.
3. **Inferencia (XGBoost):** Un modelo de regresión pre-entrenado estima la remuneración salarial.
4. **Persistencia (PostgreSQL):** La predicción y el perfil se registran en una base de datos relacional en la nube para futuros análisis.

## 🛠️ Stack Tecnológico
* **Ciencia de Datos:** Python, Scikit-Learn, Pandas, NumPy, XGBoost.
* **Ingeniería de Software:** FastAPI, Pydantic, Uvicorn.
* **Base de Datos:** PostgreSQL, Psycopg2.
* **Despliegue & DevOps:** Docker, Google Cloud Run (Serverless).

## Ejecución del proyecto localmente

### Requisitos previos
* Docker.
* Una cuenta gratuita en Neon.tech (o cualquier servidor PostgreSQL local/cloud).

### Instalación
1. Clona este repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/api-sueldos-it.git](https://github.com/TU_USUARIO/api-sueldos-it.git)
   cd api-sueldos-it