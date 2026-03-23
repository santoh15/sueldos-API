#  IT Salary Predictor - Argentina (2026.03)

A Machine Learning application designed to predict Information Technology (IT) salaries in Argentina. The prediction engine is powered by an **XGBoost** model, trained on the official **Sysarmy** survey dataset (2026.03 edition).

 **Live Demo:** [https://api-sueldos-873271753459.us-central1.run.app/](https://api-sueldos-873271753459.us-central1.run.app/)

## Tech Stack

* **Machine Learning:** Python, Pandas, NumPy, XGBoost, Scikit-learn
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Vanilla HTML, CSS, JavaScript
* **Database:** PostgreSQL
* **Deployment:** Docker, Google Cloud Plataform

## Features & Architecture

The model estimates salaries based on 12 key features:
* Age
* Total years of experience
* Years in the current role
* Years at the current company
* Number of direct reports
* Bonus reception
* Inflation adjustments in the last semester
* Seniority level
* Current role / Title
* Main programming languages and technologies
* Salary tied to USD

Additionally, the application securely stores prediction history and (optional) real salary inputs in a PostgreSQL database to monitor model drift and gather future training data.

## How to Run Locally (Docker)

The easiest and most reliable way to run this application is using Docker.

### Prerequisites
* Docker Desktop installed.
* A `.env` file in the root directory with your database credentials:
    ```env
    DATABASE_URL=postgresql://user:password@host:port/dbname
    ```

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/santoh15/sueldos-API](https://github.com/santoh15/sueldos-API)
   cd your-repo-name
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t api-sueldos .
   ```

3. **Run the container:**
   ```bash
   docker run -p 8080:8080 --env-file .env api-sueldos
   ```

4. **Access the application:**
   * **Web Interface:** `http://localhost:8080`
   * **Interactive API Docs (Swagger UI):** `http://localhost:8080/docs`

## API Endpoints

* `GET /`: Serves the main HTML frontend interface.
* `POST /predict`: Accepts a JSON payload with user data, runs the XGBoost prediction, saves the record to the database, and returns the estimated salary in ARS.
