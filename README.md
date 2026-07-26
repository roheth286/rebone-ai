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
4. **Class Balancing (SMOTE)**: Due to the severe class imbalance (only 2% fractures in the raw data), **SMOTE (Synthetic Minority Over-sampling Technique)** is applied directly to the training split to synthetically generate minority samples, balancing the positive class ratio up to **25%** for model training. *(Note: Our best-performing model, the Isolation Forest, bypasses SMOTE entirely, as it uses semi-supervised training on a clean cohort instead).*
5. **Clean Cohort Filtering (Isolation Forest Only)**: To train the Isolation Forest, we filter the training set to only include healthy patients (`Fracture = 0`) and remove borderline cases (diagnosed osteoporosis or severe T-scores $<-2.0$). This establishes an extremely pure, strong-boned baseline, causing any bone-density anomalies (fracture cases) to stand out immediately. This data purification is the primary driver behind the model's high ROC-AUC (which reaches **0.71** under Stratified 5-Fold Cross-Validation) and Precision.
6. **Weighted Feature Duplication (Isolation Forest Only)**: The model is trained on **12 distinct clinical features** (demographics, risk flags, and BMD T-scores). To emphasize the most critical clinical signals, **5 high-impact features** (`VT`, `VD`, `OP`, `Calcitriol`, `Calcitonin`) are duplicated 2 times each (yielding 10 duplicates). This expands the training matrix to **22 features in total**. Because the Isolation Forest builds trees by selecting features at random, this duplication makes the tree-builder 3 times more likely to split on these high-impact columns, focusing the model's partitions on the most predictive clinical signals.

---

## 📊 Feature Selection & Relationship Analysis

To justify selecting the **12 key features** (excluding the target `Fracture`) out of the 39 available features, we ran a correlation analysis of all 39 raw features against `Fracture` in `UA.csv`. 

You can run this analysis directly using the script in the repository:
```bash
python src/evaluation/feature_relationship_analysis.py
```

### Feature Correlation Rankings
Our 12 selected features are highlighted below, showing that they hold the highest predictive signals in the dataset while the remaining 27 columns represent low-signal noise:

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

*\*Note on Ranks 4 and 8: Ranks 4 (`TL`) and 8 (`FN`) are raw bone mineral density values of the hip and neck. We excluded them to avoid redundancy, as their corresponding T-scores (`TLT` and `FNT`) are more standardized and already included.*

### Comparison Summary
* **Average Absolute Correlation of [CHOSEN] Features**: **`0.070981`**
* **Average Absolute Correlation of Noise Features**: **`0.025768`**
* On average, our chosen features have **`2.8x` stronger relationships** with the target than the remaining 27 noise features in the dataset (such as liver enzymes ALT/AST, kidney function CREA/BUN, and other chemistry markers).

---

## 📈 Models & Baseline Results
When evaluated on the raw `UA.csv` dataset (with a 30% test split), the baseline performance of the standard models is summarized below:

| Model | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | `0.5364` | `0.2222` | **`1.0000`** | `0.3636` |
| **Random Forest** | `0.5975` | `0.4444` | `0.1379` | `0.2105` |
| **XGBoost** | `0.6120` | `0.5556` | `0.0633` | `0.1136` |
| **LightGBM** | `0.5948` | `0.2222` | `0.5000` | `0.3077` |
| **CatBoost** | `0.5973` | `0.4444` | `0.1379` | `0.2105` |
| **Isolation Forest (Ours)** | **`0.6434`** | `0.2222` | `0.4000` | `0.2857` |

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

### Key Experimental Insights & CV Loop Integration:
* **Cross-Validation vs. Single Split**: While the model achieves an ROC-AUC of **`0.6434`** on the single 30% test split, evaluating it via our **Stratified 5-Fold Cross-Validation loop** yields a much more stable and realistic ROC-AUC of **`0.7102`** (Baseline) and **`0.7247`** (Weighted). The cross-validation loop runs across the entire dataset, smoothing out the noise and split volatility caused by having only 9 positive cases in a single test split.
* **Feature Selection Boost**: Restricting the Isolation Forest to our 12 clinical features reduced dimensionality noise, causing Precision to jump from **`49.1%` to `68.9%`**.
* **Ensemble Outliers (Method 3)**: Combining Isolation Forest and One-Class SVM caught different types of anomalies, boosting Recall to **`39.52%`** while maintaining a high Precision of **`54.34%`**.
* **Weighted Tree Splits (Method 4)**: Duplicating high-impact features (`VT`, `VD`, `OP`) forced the tree-builder to split on them more frequently, yielding the highest overall ROC-AUC (**`0.7247`**).
* **Recall Maximization (Method 1)**: Tuning the decision threshold to prioritize the $F_2$-score doubled baseline Recall to **`55.24%`** (Precision: `16.20%`).

---

## 🔮 Future Outlook & Development Roadmap
1. **Development Stage**: The project is currently in the active R&D and experimentation phase.
2. **Upcoming Dataset**: We are expecting a new, larger clinical dataset with a higher proportion of positive fracture cases. This will resolve the extreme class imbalance and allow us to train more complex models.
3. **API Development**: Once the model architecture is finalized on the new dataset, we will develop a robust REST API to serve real-time fracture risk predictions for clinical applications.
