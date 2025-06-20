# Aviation Incident Analysis: Results and Methodology

## 1. Introduction
This document provides a comprehensive, data scientist-level summary of the methodology, pipeline, and key findings from our research on post-takeoff aviation incident analysis using advanced machine learning and NLP techniques.

---

## 2. Data Sources & Integration
- **Datasets:**
  - Airplane Crashes and Fatalities Since 1908
  - Archive Datasets (multiple CSVs)
  - Fatality and location reference data
- **Scope:** 17,768 incidents, 1908–2009, global coverage
- **Integration:**
  - Dynamic loading of all CSVs in `data/`
  - Encoding fixes (UTF-8, Latin-1)
  - Deduplication and data type validation

## 3. Data Cleaning & Feature Engineering
- **Missing Data:** Quantified, visualized, and handled missingness
- **Date Handling:** Unified date columns, extracted year, filtered invalid dates
- **Fatality Calculation:** Summed all fatality-related columns to create `Total_Fatalities`
- **Flight Type Classification:** NLP and keyword-based rules to assign Commercial, Military, Private
- **Text Preprocessing:**
  - Cleaned summaries (lowercase, remove numbers/punctuation, stemming, stopword removal)
  - Created `Summary_Clean` for NLP analysis

## 4. NLP-Based Cause Extraction
- **Keyword Dictionaries:** For each major cause (Mechanical/Engine Failure, Weather, Human Factor, etc.)
- **Enhanced Logic:** Pattern recognition for fallback categorization
- **Output:** Multi-label `Crash_Causes` column for each incident

## 5. Data Visualization
- **Missing Data Barplot:** `missing_data.png`
- **Flight Type Distribution:** `flight_type_distribution.png`
- **Crash Cause Distribution:** `crash_cause_distribution.png`
- **Cause Co-occurrence Heatmap & Network:** `cause_cooccurrence_heatmap.png`, `cause_cooccurrence_network.png`
- **Comprehensive Dashboard:** 9-panel commercial flight dashboard (`commercial_flight_dashboard.png`)
- **Model Metrics:** Confusion matrices, accuracy barplots

## 6. Machine Learning Pipeline
- **Task:** Binary classification (Severe [Fatalities ≥ 1] vs. Not Severe)
- **Features:** Year, Aboard, Operator, Type, Flight Type (encoded)
- **Preprocessing:**
  - Label encoding for categorical features
  - Train/test split (stratified)
- **Models:**
  - Random Forest (n=100, default params)
  - XGBoost (default params, logloss eval)
- **Evaluation:**
  - Accuracy, F1, classification report, confusion matrix
  - All metrics and plots saved in `visualizations/`

## 7. Key Results
- **Random Forest Accuracy:** 98.8%
- **XGBoost Accuracy:** 98.5%
- **High precision/recall for "Severe" class (class imbalance noted)
- **All outputs and plots are saved for review.**

## 8. Limitations & Future Work
- **Class Imbalance:** Most incidents are "Severe"; future work should address this with resampling or alternative metrics
- **Multi-Class Prediction:** Current pipeline is binary; multi-class cause prediction will be explored as more data becomes available
- **Temporal Scope:** Data limited to 2009; integrating modern datasets is a priority
- **Feature Expansion:** More technical and operational features could improve model interpretability and accuracy

## 9. Reproducibility & Best Practices
- **Script:** All steps automated in `plane_crash_analysis.py`

- **Requirements:**
  ```
  pandas>=1.5.0
  numpy>=1.21.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  networkx>=3.0
  nltk>=3.7
  scikit-learn>=1.1.0
  xgboost>=1.6.0
  lightgbm>=3.3.0
  ```
- **Best Practices:**
  - Modular, reproducible, and version-controlled code
  - Transparent data lineage and transformation steps
  - Publication-ready visualizations
  - Feature importances and model interpretability tools available

---

This document provides a detailed, professional summary of our research methodology and results. For further technical details, consult the code and data files in this repository.
LIKHITH S
