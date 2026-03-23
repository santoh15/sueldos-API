# API de Predicción Salarial IT (End-to-End ML)

Un servicio web End-to-End que predice sueldos del sector tecnológico en Argentina mediante Machine Learning. El proyecto abarca desde el entrenamiento del modelo hasta su despliegue en la nube y el almacenamiento de inferencias en tiempo real.

**[Probá la API acá](https://api-sueldos-873271753459.us-central1.run.app/)**

## Arquitectura del Sistema
1. **Frontend / Cliente:** Interfaz HTML/JS servida directamente desde la API para ingresar los 17 parámetros del usuario.
2. **Procesamiento (FastAPI):** Recibe los datos, limpia las cadenas de texto y aplica One-Hot Encoding dinámico utilizando Pandas.
3. **Inferencia (XGBoost):** Un modelo de regresión pre-entrenado estima la remuneración salarial.
4. **Persistencia (PostgreSQL):** La predicción y el perfil se registran en una base de datos relacional en la nube para futuros análisis.

##  Stack Tecnológico
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

2. Crea un archivo `.env` en la raíz con tu URL de base de datos:
   ```
   DATABASE_URL=postgresql://tu_usuario:tu_password@tu_host:5432/tu_db
   ```

3. Ejecuta el notebook `main3.ipynb` para entrenar el modelo y generar `model_xgb.pkl` y `columns.pkl` en `api_deployment/`.

### Despliegue con Docker

1. Construye la imagen:
   ```bash
   docker build -t sueldos-api .
   ```

2. Ejecuta el contenedor (inyecta la variable de entorno en runtime):
   ```bash
   docker run -p 8080:8080 -e DATABASE_URL=tu_url_de_db sueldos-api
   ```

3. Accede a la API en `http://localhost:8080`.

### Despliegue en Producción (Google Cloud Run)

1. Construye y sube la imagen a Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/TU_PROYECTO/sueldos-api
   ```

2. Despliega en Cloud Run:
   ```bash
   gcloud run deploy sueldos-api --image gcr.io/TU_PROYECTO/sueldos-api --platform managed --port 8080 --set-env-vars DATABASE_URL=tu_url_de_db
   ```