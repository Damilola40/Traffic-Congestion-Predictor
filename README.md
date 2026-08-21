# 🚦 Lagos Traffic Congestion Predictor

An end-to-end machine learning system for predicting road traffic congestion across major transit corridors in Lagos, Nigeria. The project uses **temporal, spatial, and meteorological features** to classify traffic conditions as either normal or congested.

## 📌 Project Overview

Urban traffic prediction is challenging because normal traffic observations significantly outnumber severe congestion events. This creates a **class imbalance problem**, making accuracy alone an unreliable measure of model performance.

This project formulates congestion prediction as a **binary classification problem**:

* **`0` — Normal Flow:** Free-flowing or low-density traffic conditions.
* **`1` — Congested Flow:** Moderate to severe traffic congestion, consolidated from the original `medium` and `high` traffic categories.

The primary objective is to develop a model that can identify congestion events while minimizing false congestion alerts.

## 📊 Dataset & Collection Methodology

### Data Collection

The dataset was manually collected across active observation windows over a **one-week period**, covering both peak and off-peak traffic conditions.

| Property           | Details                  |
| ------------------ | ------------------------ |
| Total observations | 248                      |
| Location           | Lagos, Nigeria           |
| Collection period  | 1 week                   |
| Task               | Binary classification    |
| Target             | Traffic congestion level |
| Train/Test split   | 80/20 stratified         |
| Test samples       | 50                       |

### 🛣️ Corridors Covered

The dataset covers major arterial routes across Lagos, including:

* **Third Mainland Bridge** — Oworonshole–Adekunle
* **Lekki-Epe Expressway** — Lekki Phase 1–Ajah
* **Ikorodu Road**
* **Lagos-Ibadan Expressway** — Ojota–Berger
* **Apapa-Oshodi Expressway** — Mile 2–Oshodi
* **Lagos-Badagry Expressway** — Mile 2–Festac
* **Funsho Williams Avenue**
* **Lagos-Abeokuta Expressway**

### 🔎 Features

The dataset contains three major categories of predictive features.

**Temporal**

* Observation hour (`obs_hour`)
* Day of the week
* Time-related traffic patterns

**Meteorological**

* Weather condition
* Temperature (°C)
* Rain probability (`rain_chance`)

**Spatial / Contextual**

* Road name
* Corridor section
* Congestion location type
* Traffic pattern

## 🛠️ Data Preprocessing & Feature Engineering

The project uses a `scikit-learn` **`ColumnTransformer` pipeline** to ensure that preprocessing is applied consistently during training, cross-validation, and inference.

### Preprocessing Workflow

1. Load and inspect the raw traffic dataset.
2. Clean missing and inconsistent values.
3. Convert the original traffic categories into a binary target.
4. Extract useful temporal features.
5. Separate numerical and categorical features.
6. Apply numerical preprocessing to continuous variables.
7. Apply `OneHotEncoder` to categorical variables.
8. Train the classifier within the complete preprocessing pipeline.
9. Evaluate using stratified cross-validation and a held-out test set.
10. Serialize the complete pipeline for future inference.

Original categorical variables such as `day`, `traffic_pattern`, and `congestion_location` are retained and encoded by the preprocessing pipeline rather than relying on manually generated encoded duplicates.

This helps keep the feature-processing workflow consistent and reduces unnecessary duplication of the same information.

## 🤖 Models

Two classification approaches were evaluated.

### 1. Logistic Regression

A **class-weighted Logistic Regression** model was used as the baseline.

Logistic Regression provides an interpretable benchmark for evaluating how well a relatively simple linear classifier can distinguish normal traffic from congestion.

### 2. XGBoost

A **tuned XGBoost classifier** was used as the primary machine learning model.

XGBoost was selected because it can capture nonlinear relationships and interactions between temporal, spatial, and meteorological features.

## 📈 Model Performance & Evaluation

Both models were evaluated using an **80/20 stratified train-test split** and **5-Fold Stratified Cross-Validation**.

The held-out test set contained:

* **42 Normal observations**
* **8 Congested observations**

Because the dataset is imbalanced, the evaluation focuses on **Macro F1, Balanced Accuracy, precision, recall, and confusion matrices** rather than accuracy alone.

### Model Comparison

| Metric                  | Logistic Regression |       **XGBoost** |
| ----------------------- | ------------------: | ----------------: |
| **5-Fold CV Macro F1**  |       0.700 ± 0.055 | **0.737 ± 0.032** |
| **Test Accuracy**       |               74.0% |         **86.0%** |
| **Balanced Accuracy**   |               69.3% |         **76.5%** |
| **Congested Precision** |                 33% |           **56%** |
| **Congested Recall**    |                 62% |           **62%** |
| **Congested F1-Score**  |                0.43 |          **0.59** |
| **Normal F1-Score**     |                0.83 |          **0.92** |

### 🏆 Key Findings

The tuned **XGBoost model outperformed the Logistic Regression baseline across the main evaluation metrics**.

XGBoost achieved:

* **86% test accuracy**
* **76.5% balanced accuracy**
* **0.737 ± 0.032 5-Fold CV Macro F1**
* **0.59 F1-score for congested traffic**
* **56% precision for congestion predictions**
* **62% recall for congestion events**

The model correctly identified **5 of the 8 congestion events** in the held-out test set while producing only **4 false congestion alerts**.

Compared with Logistic Regression, XGBoost improved:

* Test accuracy by **12 percentage points**
* Balanced accuracy by approximately **7.2 percentage points**
* Congested-class F1 from **0.43 to 0.59**
* Congested-class precision from **33% to 56%**

## 🔲 Confusion Matrix Analysis

### Logistic Regression

```text
[[32 10]
 [ 3  5]]
```

The Logistic Regression model correctly identified:

* **32 Normal** observations
* **5 Congested** observations

It produced **10 false positives**, where normal traffic was classified as congested.

### XGBoost

```text
[[38  4]
 [ 3  5]]
```

XGBoost correctly identified:

* **38 Normal** observations
* **5 Congested** observations

It reduced false positives from **10 to 4** while maintaining the same congestion recall as Logistic Regression.

This indicates that XGBoost provides a better balance between identifying congestion and avoiding unnecessary alerts.

---

## 📋 XGBoost Classification Report

```text
               precision    recall  f1-score   support

Normal (0)        0.93      0.90      0.92        42
Congested (1)     0.56      0.62      0.59         8

accuracy                              0.86        50
macro avg         0.74      0.76      0.75        50
weighted avg      0.87      0.86      0.86        50
```

The model performs strongly on the Normal class, achieving a **0.92 F1-score**.

Congestion detection remains more challenging. The model identifies **62% of congestion events**, while its **56% precision** means that slightly more than half of its congestion predictions are correct.

Given the limited number of congestion observations, these results should be interpreted cautiously.

---

## 💡 Why These Metrics Matter

A traffic prediction model can achieve high accuracy simply by predicting the majority class when congestion events are rare.

For example, a model that almost always predicts **Normal** could still appear accurate while failing to detect important congestion events.

Therefore, this project emphasizes:

* **Macro F1** — measures performance across both classes more evenly.
* **Balanced Accuracy** — accounts for the imbalance between normal and congested observations.
* **Congested Precision** — measures how reliable congestion predictions are.
* **Congested Recall** — measures how many actual congestion events are detected.
* **Confusion Matrix** — shows the types of correct and incorrect predictions.

## ⚠️ Limitations

This project is currently a **proof-of-concept** rather than a production-ready traffic prediction system.

Key limitations include:

* The dataset contains only **248 observations**.
* Data was collected manually over approximately **one week**.
* Only **8 congestion observations** are present in the held-out test set.
* The dataset may not represent long-term traffic patterns across Lagos.
* The limited number of congestion events makes minority-class metrics sensitive to individual predictions.
* Congestion detection precision still has room for improvement.
* The model predicts a binary congestion state rather than exact traffic speed, density, or travel time.
* Real-time traffic feeds are not currently integrated.

Consequently, the reported results demonstrate the feasibility of the machine learning approach but should not be interpreted as production-level traffic forecasting performance.

## 🔮 Future Improvements

Future development will focus on improving both the **data** and the prediction system.

Potential improvements include:

* Collecting substantially more traffic observations.
* Extending data collection across several months.
* Increasing the number of observed congestion events.
* Integrating real-time traffic and GPS-based data.
* Incorporating historical traffic patterns.
* Adding public holidays, accidents, road closures, and major events as features.
* Exploring additional classification algorithms.
* Experimenting with class-imbalance techniques.
* Performing more extensive hyperparameter optimization.
* Optimizing the classification threshold for the desired precision-recall trade-off.
* Building a real-time prediction API.
* Developing an interactive traffic monitoring dashboard.
* Deploying the model as a web or mobile application.

A larger and more representative dataset is expected to provide a stronger foundation for reliable congestion prediction than simply increasing model complexity.

## 📁 Repository Structure

```text
Lagos-Urban-Traffic-Predictor/
│
├── traffic data.csv
│   └── Raw manually collected traffic dataset
│
├── cleaned_traffic_data.csv
│   └── Cleaned and processed dataset
│
├── train_model.ipynb
│   └── Data cleaning, EDA, preprocessing, training,
│       evaluation, and model comparison
│
├── traffic_binary_lr_pipeline.pkl
│   └── Serialized Logistic Regression pipeline
│
├── traffic_binary_xgb_pipeline.pkl
│   └── Serialized XGBoost pipeline
│
└── README.md
    └── Project documentation
```

## 🧰 Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computation
* **Scikit-learn** — Preprocessing, pipelines, cross-validation, and Logistic Regression
* **XGBoost** — Gradient-boosted classification
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Jupyter Notebook** — Model development and experimentation

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Lagos-Urban-Traffic-Predictor
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook train_model.ipynb
```

The notebook contains the complete workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Model Training
      ↓
Cross-Validation
      ↓
Test Evaluation
      ↓
Model Comparison
```

## 🎯 Project Objective

The long-term goal is to develop a data-driven traffic intelligence system capable of helping **commuters, transport operators, and urban planners** anticipate congestion and make better transportation decisions.

This project provides a foundation for progressing from **offline traffic classification** toward **real-time traffic prediction and intelligent transportation systems** in Lagos and other rapidly growing Nigerian cities.

## 👨‍💻 Author

**Muhammad Akinwunmi Misbahudeen**

Mechatronics Engineering | AI/ML
