# 📊 Bank Marketing Term Deposit Prediction

## Data Mining Project — Classification with SMOTE & Feature Selection

---

## 🗂️ Project Overview

This project applies a full **data mining pipeline** to the **Bank Marketing dataset** (`bank-full.csv`) to predict whether a client will **subscribe to a term deposit** (binary target: `yes` / `no`). The pipeline covers every stage from raw data exploration to model evaluation and comparison.

---

## 📁 Files in This Project

| File | Description |
|------|-------------|
| `bank-full.csv` | The UCI Bank Marketing dataset (45,211 records, 17 columns) |
| `project-1.ipynb` | The main Jupyter Notebook with all analysis and model code |

---

## 🧾 Dataset Description

**Source:** UCI Machine Learning Repository — Bank Marketing Dataset  
**Separator:** Semicolon (`;`)  
**Shape:** 45,211 rows × 17 columns  
**Missing Values:** None

### Features

| # | Column | Type | Description |
|---|--------|------|-------------|
| 0 | `age` | int | Client's age |
| 1 | `job` | object | Type of job (e.g., management, technician, blue-collar) |
| 2 | `marital` | object | Marital status (married, single, divorced) |
| 3 | `education` | object | Education level (primary, secondary, tertiary, unknown) |
| 4 | `default` | object | Has credit in default? (yes/no) |
| 5 | `balance` | int | Average yearly balance in euros |
| 6 | `housing` | object | Has housing loan? (yes/no) |
| 7 | `loan` | object | Has personal loan? (yes/no) |
| 8 | `contact` | object | Contact communication type (cellular, telephone, unknown) |
| 9 | `day` | int | Last contact day of the month |
| 10 | `month` | object | Last contact month of the year |
| 11 | `duration` | int | Last contact duration in seconds |
| 12 | `campaign` | int | Number of contacts during this campaign |
| 13 | `pdays` | int | Days since client was last contacted (-1 = not contacted) |
| 14 | `previous` | int | Number of contacts before this campaign |
| 15 | `poutcome` | object | Outcome of the previous marketing campaign |
| 16 | `y` | object | **Target:** Has the client subscribed? (yes/no) |

---

## 🔬 Methodology — Step-by-Step Pipeline

### Step 1: Data Loading & Initial Exploration
- Loaded `bank-full.csv` using `pandas` with semicolon separator.
- Verified shape: **45,211 rows × 17 columns**.
- Confirmed **zero missing values** across all columns.

---

### Step 2: Exploratory Data Analysis (EDA)
- Plotted the **distribution of the target variable** (`y`).
- **Key Finding:** Severe class imbalance detected:
  - `no` (did not subscribe): **~88%** of records
  - `yes` (subscribed): **~12%** of records
- This imbalance means that naive models would simply predict "no" and achieve high accuracy — making **SMOTE** essential.

---

### Step 3: Data Preprocessing

#### Feature/Target Split
```python
X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
```

#### Feature Types
- **Numerical:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- **Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

#### Preprocessing Pipeline (`ColumnTransformer`)
| Step | Transformer | Applied To |
|------|-------------|------------|
| Scaling | `MinMaxScaler` | Numerical features |
| Encoding | `OneHotEncoder(handle_unknown='ignore')` | Categorical features |

#### Train/Test Split
- **80%** training / **20%** testing
- `stratify=y` used to maintain class proportions
- `random_state=42` for reproducibility

---

### Step 4: Feature Selection (Chi-Squared Test)
- Applied `SelectKBest` with `chi2` scoring to select the **top 20 features** from the preprocessed (encoded + scaled) feature space.
- **Selected Features:**
  ```
  duration, pdays, job_blue-collar, job_retired, job_student,
  marital_single, education_tertiary, housing_no, housing_yes,
  loan_yes, contact_cellular, contact_unknown, month_apr,
  month_dec, month_mar, month_may, month_oct, month_sep,
  poutcome_success, poutcome_unknown
  ```
- This reduces noise and training time by cutting from 50+ encoded features to only the 20 most statistically relevant ones.

---

### Step 5: Handling Class Imbalance with SMOTE

**Before SMOTE:**
| Class | Count |
|-------|-------|
| 0 (No) | 31,937 |
| 1 (Yes) | 4,231 |

**After SMOTE (applied to training data only):**
| Class | Count |
|-------|-------|
| 0 (No) | 31,937 |
| 1 (Yes) | 31,937 |

> SMOTE (Synthetic Minority Over-sampling Technique) generates **synthetic samples** for the minority class rather than simply duplicating existing ones. This prevents the model from being biased toward the majority "no" class.

---

### Step 6: Model Training & Evaluation

Three classifiers were trained on the SMOTE-balanced training data and evaluated on the **original (unbalanced) test set**.

#### Model 1: Logistic Regression

```python
LogisticRegression(random_state=42, max_iter=1000)
```

| Metric | Class 0 (No) | Class 1 (Yes) |
|--------|--------------|----------------|
| Precision | 0.97 | 0.41 |
| Recall | 0.85 | 0.80 |
| F1-Score | 0.91 | 0.54 |
| **Overall Accuracy** | | **84%** |

---

#### Model 2: Decision Tree Classifier

```python
DecisionTreeClassifier(random_state=42)
```

| Metric | Class 0 (No) | Class 1 (Yes) |
|--------|--------------|----------------|
| Precision | 0.95 | 0.38 |
| Recall | 0.86 | 0.64 |
| F1-Score | 0.90 | 0.48 |
| **Overall Accuracy** | | **84%** |

---

#### Model 3: Support Vector Machine (SVM)

```python
SVC(kernel='rbf', probability=True, random_state=42)
```

| Metric | Class 0 (No) | Class 1 (Yes) |
|--------|--------------|----------------|
| Precision | 0.97 | 0.40 |
| Recall | 0.84 | 0.81 |
| F1-Score | 0.90 | 0.54 |
| **Overall Accuracy** | | **84%** |

---

### Step 7: Final Model Comparison

Two final visualizations compare all three models:

1. **Bar Chart** — Side-by-side comparison of Accuracy, Precision, Recall, and F1-Score for the positive class (`yes`).
2. **ROC Curve** — Shows each model's true positive rate vs. false positive rate across all thresholds.

---

## 🧰 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical data visualization |
| `sklearn` | Preprocessing, feature selection, models, and metrics |
| `imblearn` | SMOTE for class imbalance handling |

---

## 🏁 Key Takeaways

- **Class imbalance is severe** (~88% "no" vs. ~12% "yes") and must be addressed with SMOTE.
- **Feature selection** (top 20 via Chi-squared) reduces dimensionality and noise effectively.
- All three models achieve **~84% overall accuracy**, but differ in **recall for the positive class**:
  - **Logistic Regression & SVM** achieve the highest recall for "yes" (~80–81%), making them better choices if the goal is to **not miss potential subscribers**.
  - **Decision Tree** has lower recall (64%) but similar overall accuracy.
- `duration` (call length) and `poutcome_success` (previous campaign success) are among the most predictive features.

---

## ▶️ How to Run

1. Ensure both files (`bank-full.csv` and `project-1.ipynb`) are in the **same directory**.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
3. Open and run all cells in `project-1.ipynb` (e.g., via **Jupyter Notebook** or **VS Code**).

---

*Project completed as part of a Data Mining course.*
