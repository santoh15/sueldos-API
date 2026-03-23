#  IT Salary Forecast Argentina V1.0 (Sysarmy 2026.03)

This project consists of an end-to-end Machine Learning pipeline and web API for predicting Information Technology (IT) salaries in Argentina. The model leverages variables such as years of experience, seniority, job role, and technologies used, based on the official Sysarmy survey.

The main idea of the whole project is to build a robust, modular **MLOps pipeline**, bridging the gap between a Data Science experiment and a production-ready application. It ranges from data ingestion and model training to containerized deployment and database logging for future retraining.

**Live Demo:** [https://api-sueldos-873271753459.us-central1.run.app/](https://api-sueldos-873271753459.us-central1.run.app/)

## Project Structure

The project is structured in the following way to separate data science logic from deployment:

- `main.ipynb`: Jupyter notebook where we load the raw data, train the model, evaluate metrics, and visualize the results.
- `src/`: Contains all the modular Data Science code used in the project:
  - `EDA.py`: Performs exploratory data analysis, missing data handling, and outlier removal using quantile heuristics.
  - `encoding.py`: Custom feature transformations, including a specific one-hot encoder for multiselect variables (e.g., programming languages separated by commas).
  - `train.py`: Model training, hyperparameter tuning, and performance metrics calculation.
- `api_deployment/`: Contains the backend logic for the production environment:
  - `main.py`: FastAPI application setup and database initialization.
  - `predict.py`: Exposes the `/predict` endpoint and processes incoming JSON payloads through the saved pipeline.
  - `interface.py` & `index.html`: Serves the frontend user interface.
- `Dockerfile` & `requirements.txt`: Containerization instructions and dependencies to ensure reproducibility.

##  Intended Features

- Load and clean the dataset, removing extreme outliers using quantile thresholds.
- Transform features: dynamic one-hot encoding for categorical variables and custom parsing for comma-separated technology stacks.
- Target transformation: Log-transform for the target variable (Salary) to handle heavy right-skewness.
- Train a robust predictive model using **XGBoost**.
- Evaluate the model using MAE and R2 Score.
- Report 95% confidence intervals for the metrics using bootstrap resampling.
- REST API deployment using FastAPI and Docker (hosted on Google Cloud Run).
- Data Flywheel: Save model predictions and optional real-salary inputs in a PostgreSQL database to monitor model drift.

## Modeling Approach

The first step was to explore the dataset and handle obvious issues like null values and extreme outliers. Since salary data is highly prone to extreme values (both unusually low entries and executive-level salaries), I applied quantile-based filtering to remove them. This ensures the model learns the general market trend rather than overfitting to edge cases.

With the data cleaned, I inspected the target variable (Salary). As expected in economics data, it was heavily skewed towards higher salaries. I applied a logarithmic transformation (`np.log1p`) to reduce that skewness and approximate a normal distribution, which significantly helps the model's stability. 

For the features, categorical variables like "Seniority" and "Job Title" were one-hot encoded. A special custom function was developed for the "Technologies" column, since respondents could select multiple languages (e.g., "python, sql, aws"). The function splits these strings and creates individual binary columns for each technology, capturing the premium paid for specific tech stacks.

The model chosen was **XGBoost**. The motivation is its outstanding performance with tabular data, handling of sparse matrices (created by our heavy one-hot encoding), and capability to capture non-linear relationships without massive hyperparameter tuning.

For evaluation, I used MAE and R2 on the real salary values (after reversing the log with `np.expm1`). I also added 95% confidence intervals using bootstrap sampling to get a rough idea of how stable the model is across different data subsets.

## 🔄 MLOps & Additional Features

- **Database Logging:** Every prediction request is saved into a PostgreSQL database hosted on Neon.tech. The UI also includes an optional "Real Salary" field. This allows us to collect ground-truth data to monitor model drift and easily retrain the model in the future.
- **Dockerized Environment:** The whole API is containerized, solving the "it works on my machine" problem and allowing seamless serverless deployment on Google Cloud Run.

## 🚀 How to run the project

Clone this repository:

```bash
git clone [https://github.com/TU_USUARIO/api-sueldos-it.git](https://github.com/TU_USUARIO/api-sueldos-it.git)
cd api-sueldos-it
```

**To run the API locally using Docker:**

1. Create a `.env` file in the root directory with your database URL:
```env
DATABASE_URL=postgresql://user:password@host:port/dbname
```
2. Build and run the container:
```bash
docker build -t api-sueldos .
docker run -p 8080:8080 --env-file .env api-sueldos
```
3. Open your browser at `http://localhost:8080` to see the UI, or `http://localhost:8080/docs` to interact with the Swagger API.

## 📡 API JSON Format for Predictions

To generate predictions programmatically, you can send a `POST` request to `/predict`. The API expects a JSON payload with the following structure:

```json
{
  "dedicacion": "Full-Time",
  "recibis_algun_tipo_de_bono": "Un sueldo",
  "tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre": 1,
  "trabajo_de": "Data Scientist",
  "anos_de_experiencia": 3.5,
  "antiguedad_en_la_empresa_actual": 2.0,
  "anos_en_el_puesto_actual": 2.0,
  "cuantas_personas_tenes_a_cargo": 0,
  "lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual": "python, sql, docker",
  "tengo_edad": 28,
  "sueldo_dolarizado": 0,
  "seniority": "Semi-Senior"
}
```

## ⚖️ Ethical Considerations

During feature selection, survey datasets often include demographic variables such as Gender. While these variables can sometimes capture existing market disparities and statistically reduce the MAE, we carefully considered the implications of deploying such a model in production.

Including gender as a predictive feature risks perpetuating and mathematically validating historical wage gaps. To ensure fairness and provide a gender-neutral salary estimation based purely on technical merit, experience, and role responsibilities, demographic bias features were neutralized in the deployed API.
