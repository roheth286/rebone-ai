# Rebone AI: Fracture Risk Assessment Pipeline (Still Under Production)

Rebone AI is a machine learning and clinical analysis pipeline designed to identify high-risk bone profiles and predict potential patient fractures using bone mineral density (BMD) data, demographics, and clinical history.

---

## 🚀 Getting Started & Execution

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ installed. First, clone the repository, activate your virtual environment, and install the project dependencies:

```bash
# Activate your virtual environment (Windows example)
.venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Running the Pipeline
The core orchestration of the preprocessing, cleaning, splitting, training, and evaluation is managed inside the Jupyter Notebook:
* **Notebook Path**: `notebooks/notebook_03.ipynb`

To run the pipeline:
1. Open the notebook via VS Code or your Jupyter environment:
   ```bash
   jupyter notebook notebooks/notebook_03.ipynb
   ```
2. Run all cells sequentially. The notebook will automatically:
   * Load and clean the raw data.
   * Split the dataset into train and test cohorts.
   * Build and fit the preprocessing pipeline.
   * Train five distinct machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost).
   * Evaluate all models and output performance metrics/plots.

---

## 📋 Problem Statement
The goal is to predict whether a patient will experience a bone fracture (`Fracture = 1`) based on their demographic variables, medical conditions, medications, and bone mineral density measurements. Early identification of high-risk patients allows for timely clinical interventions (such as prescribing bone-strengthening medications like bisphosphonates).

---

## 📊 Dataset
* **Raw Data File**: `data/raw/UA.csv`
* **Size**: 1,537 rows, 40 columns (demographics, blood chemistry, and bone measurements).

---

## 🛠️ Preprocessing Pipeline
To reduce overfitting and eliminate clinical noise, the cleaning pipeline selects **13 key clinical features** out of the 39 available columns:
* **Demographics**: `Gender` (Encoded: Male=0, Female=1), `Age`, `BMI`
* **Clinical Indicators**: `VT` (Vertebral T-Score), `VD` (Vitamin D Status), `Calcitriol`, `Calcitonin`, `OP` (Osteoporosis Diagnosis), `Calsium`, `COPD`
* **Bone Density Measurements**: `TL` (Total Hip BMD), `TLT` (Total Hip T-Score), `FN` (Femoral Neck BMD)

### Core Preprocessing Steps:
1. **Missing Value Imputation**: Empty values are imputed using the column medians.
2. **Outlier Capping**: Continuous variables (`Age`, `BMI`, `L1-4T`, `FN`, `TLT`) are capped at $1.5 \times \text{IQR}$ to prevent extreme values from distorting predictions.
3. **Feature Scaling**: Continuous columns are standardized using `StandardScaler` (mean=0, variance=1) while binary flags bypass scaling.
4. **Class Balancing (SMOTE)**: Due to the severe class imbalance (only 2% fractures in the raw data), **SMOTE (Synthetic Minority Over-sampling Technique)** is applied directly to the training split to synthetically generate minority samples, balancing the positive class ratio up to **25%** for model training.

---

## 📈 Models & Baseline Results
When evaluated on the raw `UA.csv` dataset (with a 30% test split), the baseline performance of the standard models is summarized below:

| Model | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score | Status |
|---|:---:|:---:|:---:|:---:|---|
| **Logistic Regression** | `0.5364` | `0.2222` | **`1.0000`** | `0.3636` | Missed 7 out of 9 |
| **Random Forest** | `0.5975` | `0.4444` | `0.1379` | `0.2105` | Missed 5 out of 9 |
| **XGBoost** | `0.6120` | `0.5556` | `0.0633` | `0.1136` | Missed 4 out of 9 |
| **LightGBM** | `0.5948` | `0.2222` | `0.5000` | `0.3077` | Missed 7 out of 9 |
| **CatBoost** | `0.5973` | `0.4444` | `0.1379` | `0.2105` | Missed 5 out of 9 |

---

## ⚠️ Key Challenges
Standard machine learning models struggle to achieve high predictive accuracy on this dataset due to three major bottlenecks:

1. **Extreme Class Imbalance (2% Positive Rate)**: 
   Out of 1,537 patients, **only 31** experienced a fracture. In a standard test split, there are only 6 to 9 positive cases, making evaluation metrics highly volatile (e.g., a single false positive drops precision by $15\%$).
2. **Weak Feature-Target Relationships**:
   No single feature exhibits a strong correlation with the target. Even the strongest clinical indicator (`VT`) has a Pearson correlation of only `0.2182` and high specificity but low sensitivity.
3. **The Precision/Recall Dilemma**:
   Due to the sparse signal, attempting to raise Recall (catching more fractures) forces the models to flag hundreds of healthy patients as high-risk, dropping Precision to $<10\%$. Conversely, optimizing for Precision limits predictions to a few obvious cases, dropping Recall below $25\%$.

---

## 🔬 Experimental Work: Anomaly Detection (Isolation Forest)
To address the class imbalance, we are experimenting with **Semi-Supervised Anomaly Detection** using **Isolation Forest**. Instead of training models on both classes, the model is trained **only** on healthy patients (`Fracture = 0`) to learn a "normal bone profile" and flags fractures as anomalies (outliers).

We evaluated these techniques using **Stratified 5-Fold Cross-Validation** on the raw `UA.csv` dataset:

### Experimental Performance Summary:

| Configuration | ROC-AUC | Precision | Recall | F1-Score | F2-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (IForest, F1-Thresh)** | `0.7204` | **`0.6892`** | `0.2619` | `0.2539` | `0.2172` |
| **Method 1 (F2-Thresh)** *[High Recall]* | `0.7204` | `0.1620` | **`0.5524`** | `0.2106` | `0.3056` |
| **Method 3 (SVM Union)** *[Ensemble]* | `0.7140` | `0.5434` | **`0.3952`** | `0.2607` | `0.2732` |
| **Method 4 (Weighted Features)** *[Balanced]* | **`0.7247`** | `0.5450` | **`0.3619`** | **`0.2745`** | `0.2787` |
| **Comb D (F2-Thresh + Clean + SVM)** | `0.7205` | `0.1619` | `0.5190` | `0.2287` | **`0.3282`** |

### Key Experimental Insights:
* **Feature Selection Boost**: Restricting the Isolation Forest to our 13 clinical features reduced dimensionality noise, causing Precision to jump from **`49.1%` to `68.9%`**.
* **Ensemble Outliers (Method 3)**: Combining Isolation Forest and One-Class SVM caught different types of anomalies, boosting Recall to **`39.52%`** while maintaining a high Precision of **`54.34%`**.
* **Weighted Tree Splits (Method 4)**: Duplicating high-impact features (`VT`, `VD`, `OP`) forced the tree-builder to split on them more frequently, yielding the highest overall ROC-AUC (**`0.7247`**).
* **Recall Maximization (Method 1)**: Tuning the decision threshold to prioritize the $F_2$-score doubled baseline Recall to **`55.24%`** (Precision: `16.20%`).

---

## 🔮 Future Outlook & Development Roadmap
1. **Development Stage**: The project is currently in the active R&D and experimentation phase.
2. **Upcoming Dataset**: We are expecting a new, larger clinical dataset with a higher proportion of positive fracture cases. This will resolve the extreme class imbalance and allow us to train more complex models.
3. **API Development**: Once the model architecture is finalized on the new dataset, we will develop a robust REST API to serve real-time fracture risk predictions for clinical applications.
