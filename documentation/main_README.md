# Comprehensive Aviation Safety Analysis for Real-Time Pilot Alerting

## 1. Project Overview

### 1.1. Core Purpose

This project is dedicated to enhancing aviation safety by building a sophisticated machine learning model that identifies the most probable causes of aviation accidents, with a specific focus on events occurring after the aircraft has lifted off. The ultimate goal is to create a system capable of providing **real-time, actionable alerts to pilots**, leveraging historical data and advanced analytics to prevent incidents before they escalate.

By analyzing over a century of aviation data, this project extracts and categorizes crash causes, identifies critical risk factors, and provides interpretable insights that can be translated into a live alerting system for the flight deck.

### 1.2. Key Objectives

*   **Dynamic Data Integration**: To automatically load, clean, and consolidate aviation incident data from multiple sources.
*   **NLP-Powered Cause Analysis**: To use Natural Language Processing (NLP) to analyze textual summaries of accidents and automatically categorize the primary cause of each incident.
*   **Predictive Modeling**: To build and train a highly accurate machine learning model (XGBoost) to predict the cause of a potential incident based on a variety of features.
*   **Explainable AI (XAI)**: To use SHAP (SHapley Additive exPlanations) to interpret the model's predictions, making the results understandable and actionable.
*   **Pilot Alerting Framework**: To design a conceptual framework for deploying the model in a real-world scenario to provide live alerts to pilots.

---

## 2. Codebase Structure

The project is organized into the following directories and files:

```
.
├── .venv/                   # Virtual environment for Python
├── data/                     # All raw data files (CSVs, text files)
├── documentation/            # Contains this detailed README file
│   └── main_README.md
├── notebooks/                # Jupyter notebooks for analysis
│   ├── plane_crash_analysis.ipynb  # The main, comprehensive analysis notebook
│   └── crash_cause_prediction.ipynb # An older, deprecated notebook
├── src/                      # (Currently empty) Intended for utility scripts/modules
└── visualizations/           # (Currently empty) Intended for saved plots and charts
```

### 2.1. Key Files

*   **`notebooks/plane_crash_analysis.ipynb`**: This is the heart of the project. It contains all the code for data loading, preprocessing, NLP analysis, model training, and interpretation. All the results and visualizations are generated from this notebook.
*   **`data/*.csv`**: A collection of CSV files containing historical data on aviation accidents, fatalities, and other relevant information. The project is designed to dynamically load all CSVs in this directory.

---

## 3. Data Sources and Preprocessing

### 3.1. Dynamic Data Loading

The project avoids hardcoding file names and instead dynamically scans the `data/` directory for all available `.csv` files. It attempts to load each file, handling potential encoding issues (UTF-8 and Latin-1), and consolidates them into a single pandas DataFrame. This makes the project robust and easily adaptable to new data.

### 3.2. Data Cleaning and Feature Engineering

The consolidated dataset undergoes a rigorous cleaning and feature engineering process:
*   **Date and Time Conversion**: The date column is converted to a datetime object, and the `Year` is extracted.
*   **Fatality Calculation**: All columns related to fatalities are converted to numeric types and summed to create a `Total_Fatalities` column.
*   **NLP Preprocessing**: The textual `Summary` column is cleaned by removing punctuation, numbers, and stopwords, and by applying stemming to prepare it for NLP analysis.

---

## 4. Methodology

The analysis is conducted in a sequential manner within the `plane_crash_analysis.ipynb` notebook.

### 4.1. NLP-Based Crash Cause Categorization

A key innovation of this project is the use of NLP to automatically categorize the cause of each crash.
*   **Keyword-Based Classification**: A dictionary of keywords is used to classify the cleaned summary text into one of the following categories:
    *   `Mechanical/Engine Failure`
    *   `Weather Related`
    *   `Pilot Error/Human Factor`
    *   `Attack/Sabotage`
    *   `Fuel Issue`
    *   `Bird Strike`
    *   `Unknown/Other`
*   **New Feature Creation**: This process results in a new `Crash_Cause` column, which serves as the target variable for our predictive model.

### 4.2. Predictive Modeling

Two models are built and evaluated: a baseline model and an enhanced model.

*   **Baseline Model**:
    *   **Features**: High-level, non-technical features like `Operator`, `Type`, `Country`, `Year`, `Aboard`, and `Fatalities`.
    *   **Algorithms**: Random Forest and XGBoost classifiers are trained and compared.
    *   **Result**: XGBoost is identified as the better-performing model.

*   **Enhanced Model**:
    *   **Synthetic Feature Generation**: To simulate a real-world scenario with more technical data, the following synthetic features are generated:
        *   `Days_Since_Inspection`
        *   `Turbulence_In_gforces`
        *   `Adverse_Weather_Metric`
    *   **Result**: The enhanced model, particularly with the XGBoost algorithm, shows a significant improvement in performance, demonstrating the value of incorporating technical features.

### 4.3. Model Interpretation with Explainable AI (XAI)

A model's predictions are only useful if they can be understood. This project uses SHAP to provide deep insights into the model's behavior.

*   **SHAP Summary Plots (Bar and Dot)**: These plots show the overall feature importance, ranking the features that have the most impact on the model's predictions. The dot plot further shows how the value of a feature (high or low) affects the prediction.
*   **SHAP Dependence Plots**: These plots reveal the relationship between a single feature and the model's output for a specific class. For example, they show how an increase in `Turbulence_In_gforces` directly increases the likelihood of the model predicting a "Weather Related" cause.
*   **SHAP Force and Waterfall Plots**: These plots provide the most granular level of detail, explaining exactly how the model arrived at its prediction for a *single, specific incident*. They show how each feature's value pushed the prediction towards or away from the final outcome.
*   **SHAP Heatmap and Decision Plots**: These provide a more global view of feature impacts across many predictions, helping to identify broader patterns in the model's decision-making process.

### 4.4. Comprehensive Correlation Analysis

To uncover relationships within the data, a comprehensive correlation analysis is performed:
*   **Numerical Correlation**: A heatmap is generated to show the correlation between all numerical features.
*   **Categorical Analysis**: The relationship between key categorical features (like `Country`, `Aircraft Manufacturer`) and the `Crash_Cause` is visualized using crosstabs and heatmaps.

---

## 5. Deployment Framework for Real-Time Pilot Alerting

The insights gained from this analysis can be operationalized into a real-time pilot alerting system. Here is a conceptual framework for how this could be achieved:

### 5.1. System Architecture

1.  **Data Ingestion**: A live data feed from the aircraft's sensors and systems (e.g., flight data recorder, weather sensors, maintenance logs) would provide the input features for the model.
2.  **Model Hosting**: The trained XGBoost model would be deployed on a dedicated, high-availability server or an onboard computer.
3.  **Real-Time Prediction**: The system would continuously run the model on the incoming data stream, generating a probability distribution for each potential cause of an incident *in real-time*.
4.  **Alerting Logic**:
    *   **Thresholds**: If the probability for a specific cause (e.g., "Mechanical/Engine Failure") exceeds a predefined threshold, an alert is triggered.
    *   **SHAP-Powered Explanations**: The alert would not just be a warning; it would be accompanied by a simplified, human-readable explanation generated from the SHAP values. For example:
        > **"ALERT: High Probability of Mechanical Failure. REASON: Engine vibration is high, and days since last inspection is above average."**
5.  **Pilot Interface**: The alert would be displayed on a screen in the cockpit, providing the pilot with immediate, interpretable, and actionable information.

### 5.2. Example Alert Scenario

*   **Conditions**: An aircraft is in flight. The system detects a combination of high turbulence, a slight drop in hydraulic pressure, and the local weather report indicates a storm cell nearby.
*   **Model Prediction**: The model's output shows a high probability for both "Weather Related" and "Mechanical/Engine Failure".
*   **Alert to Pilot**:
    > **"CAUTION: Multiple risk factors detected."**
    > **1. High Probability of WEATHER-RELATED incident (due to high turbulence).**
    > **2. Moderate Probability of HYDRAULIC issue (due to pressure drop).**
    > **RECOMMENDATION: Check hydraulic systems and consider altering course to avoid storm cell."**

---

## 6. How to Use This Project

### 6.1. Environment Setup

1.  **Clone the repository**.
2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
3.  **Install the required packages**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap nltk
    ```
4.  **Download NLTK data**: Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

### 6.2. Running the Analysis

1.  **Place your data files** in the `data/` directory.
2.  **Open and run the `notebooks/plane_crash_analysis.ipynb` notebook** in a Jupyter environment (like VS Code or Jupyter Lab). The notebook is designed to be run sequentially from top to bottom.
