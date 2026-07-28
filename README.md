# Rebone AI: Clinical Machine Learning Pipeline & Production REST API

Rebone AI is an end-to-end clinical machine learning pipeline and containerized REST API designed to evaluate patient fracture risks using bone mineral density (BMD) data, demographic indicators, and clinical history.

---

## 🚀 Getting Started & Execution

### 1. Running with Docker (Recommended)
Ensure Docker and Docker Desktop are installed and running on your system.

To build and launch the containerized application microservices (FastAPI + PostgreSQL):

```bash
docker-compose up --build
```

Once the containers start up:
* **Interactive API Documentation (Swagger UI)**: Open `http://localhost:8000/docs` in your browser.
* **Database Service**: PostgreSQL 15 running on port `5432`.

### 2. Local Python Environment Execution
Alternatively, you can run the server directly within a local Python environment:

```bash
# Activate virtual environment (Windows example)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Uvicorn development server
python -m uvicorn src.api.main:app --reload --port 8000
```

---

## 🏗️ Backend System Architecture & MLOps

Rebone AI features a production-ready REST API built with FastAPI and PostgreSQL:

* **Containerized Microservices**: Orchestrated via Docker Compose linking web and database services.
* **Authentication**: Stateless JWT access tokens (`HS256`) and Google OAuth 2.0 SSO integration (`POST /api/v1/auth/google`).
* **MLOps Model Serving**: In-memory model lifespan loading (`isolation_forest.joblib`) executing a 7-step deterministic inference pipeline.
* **Database Persistence & Auditability**: Automatic logging of raw clinical inputs, calculated risk scores, prediction labels, and model versions to PostgreSQL (`prediction_records` table).

*For a deep dive into sequence diagrams, database schemas, and system design trade-offs, please refer to [ARCHITECTURE.md](file:///c:/Users/rohet/OneDrive/Documents/CS_WORK/rebone-ai/ARCHITECTURE.md).*

---

## 📋 Problem Statement
The goal is to predict whether a patient will experience a bone fracture (`Fracture = 1`) based on demographic variables, medical conditions, medications, and bone mineral density measurements. Early identification of high-risk patients allows for timely clinical interventions (such as prescribing bone-strengthening medications like bisphosphonates).

---

## 📊 Dataset
* **Raw Data File**: `data/raw/UA.csv`
* **Size**: 1,537 rows, 40 columns (demographics, blood chemistry, and bone measurements).

---

## 🛠️ Preprocessing Pipeline
To reduce overfitting and eliminate clinical noise, the cleaning pipeline selects **12 key clinical features** out of the 39 available columns:
* **Demographics**: `Gender` (Encoded: Male=0, Female=1), `Age`, `BMI`
* **Clinical Indicators**: `VT` (Vertebral T-Score), `VD` (Vitamin D Status), `Calcitriol`, `Calcitonin`, `OP` (Osteoporosis Diagnosis), `Calsium`
* **Bone Density Measurements**: `L1-4T` (L1-4 T-Score), `TLT` (Total Hip T-Score), `FNT` (Femoral Neck T-Score)

### Core Preprocessing Steps:
1. **Missing Value Imputation**: Empty values are imputed using column medians.
2. **Outlier Capping**: Continuous variables (`Age`, `BMI`, `L1-4T`, `FNT`, `TLT`) are capped at $1.5 \times \text{IQR}$.
3. **Feature Scaling**: Continuous columns are standardized using `StandardScaler` (mean=0, variance=1) while binary flags bypass scaling.
4. **Clean Cohort Filtering**: Trained strictly on healthy patient cohorts (`Fracture = 0`) to establish a pure baseline profile, allowing fracture cases to stand out as anomalies.
5. **Weighted Feature Duplication**: High-impact clinical features (`VT`, `VD`, `OP`, `Calcitriol`, `Calcitonin`) are duplicated twice (`_dup1`, `_dup2`) to expand the feature space to 22 columns, focusing tree split choices on key clinical indicators.

---

## 📊 Feature Selection & Relationship Analysis

To justify selecting the **12 key features** (excluding the target `Fracture`) out of the 39 available features, a correlation analysis was executed against `Fracture` in `UA.csv`. 

You can run this analysis directly:
```bash
python src/evaluation/feature_relationship_analysis.py
```

### Feature Correlation Rankings
Our 12 selected features hold the highest predictive signals in the dataset:

| Rank | Feature Name | Absolute Correlation | Direction | Status |
|:---:|---|:---:|:---:|---|
| 1 | **VT** | `0.218215` | Positive | [CHOSEN] |
| 2 | **VD** | `0.120031` | Positive | [CHOSEN] |
| 3 | **Calcitriol** | `0.106859` | Positive | [CHOSEN] |
| 5 | **TLT** | `0.089259` | Negative | [CHOSEN] |
| 6 | **OP** | `0.081922` | Positive | [CHOSEN] |
| 7 | **Calsium** | `0.071116` | Positive | [CHOSEN] |
| 9 | **Calcitonin** | `0.064247` | Positive | [CHOSEN] |
| 14 | **FNT** | `0.040301` | Negative | [CHOSEN] |
| 21 | **Age** | `0.023247` | Negative | [CHOSEN] |
| 25 | **BMI** | `0.015754` | Positive | [CHOSEN] |
| 30 | **L1.4T** | `0.010737` | Positive | [CHOSEN] |
| 32 | **Gender** | `0.010080` | Positive | [CHOSEN] |

---

## 📈 Models & Baseline Results
When evaluated on the raw `UA.csv` dataset (with a 30% test split), model performances are summarized below:

| Model | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | `0.5364` | `0.2222` | **`1.0000`** | `0.3636` |
| **Random Forest** | `0.5975` | `0.4444` | `0.1379` | `0.2105` |
| **XGBoost** | `0.6120` | `0.5556` | `0.0633` | `0.1136` |
| **LightGBM** | `0.5948` | `0.2222` | `0.5000` | `0.3077` |
| **CatBoost** | `0.5973` | `0.4444` | `0.1379` | `0.2105` |
| **Isolation Forest (Ours)** | **`0.6434`** | `0.2222` | `0.4000` | `0.2857` |

---

## 🔮 Future Outlook & Development Roadmap
1. **Expanded Clinical Dataset**: The system architecture is built to seamlessly ingest an upcoming expanded clinical dataset with higher positive fracture prevalence.
2. **Model Retraining & Fine-Tuning**: Re-evaluating the tree ensemble on the expanded dataset once available to further refine sensitivity thresholds.
