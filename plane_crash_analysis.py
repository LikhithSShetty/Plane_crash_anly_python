# Converted from Jupyter notebook: plane_crash_analysis.ipynb
# This script performs post-takeoff aviation incident cause analysis using real historical data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import itertools
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    # Download stopwords if not already present
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    print("🚀 POST-TAKEOFF INCIDENT CAUSE ANALYSIS")
    print("=" * 60)

    # 1. Load all available datasets from the data directory
    data_dir = 'data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        try:
            try:
                df_item = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df_item = pd.read_csv(file_path, encoding='latin1')
            if not df_item.empty:
                key = os.path.splitext(file)[0].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                dataframes[key] = df_item
                print(f"Loaded {file}: {df_item.shape[0]:,} records, {df_item.shape[1]} columns")
            else:
                print(f"Skipping empty file: {file}")
        except Exception as e:
            print(f"Could not load {file}: {e}")

    # Assign more meaningful names to the dataframes
    df_historical = dataframes.get('Airplane_Crashes_and_Fatalities_Since_1908', pd.DataFrame())
    df_fatalities = dataframes.get('Airplane_crash_fatalities', pd.DataFrame())
    df_states = dataframes.get('USState_Codes', pd.DataFrame())
    archive_keys = [key for key in dataframes.keys() if 'archive' in key]
    df_archive_list = [dataframes[key] for key in archive_keys if not dataframes[key].empty]
    if df_archive_list:
        df_archive_combined = pd.concat(df_archive_list, ignore_index=True, sort=False)
    else:
        df_archive_combined = pd.DataFrame()
    df_list = []
    if not df_historical.empty:
        df_list.append(df_historical)
    if not df_archive_combined.empty:
        df_list.append(df_archive_combined)
    if df_list:
        df = pd.concat(df_list, ignore_index=True, sort=False)
        df.drop_duplicates(inplace=True)
        print(f"Consolidated dataframe created with shape: {df.shape}")
    else:
        df = pd.DataFrame()
        print("Warning: Could not create a consolidated dataframe from primary sources.")

    # 2. Data exploration and cleaning
    if not df.empty:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_data = pd.DataFrame({'Column': missing_percentage.index, 'Missing_Percentage': missing_percentage.values}).sort_values('Missing_Percentage', ascending=False)
        print("Columns with highest missing data:")
        print(missing_data[missing_data['Missing_Percentage'] > 0].head(15))
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            primary_date_col = date_cols[0]
            df[primary_date_col] = pd.to_datetime(df[primary_date_col], errors='coerce')
            df.dropna(subset=[primary_date_col], inplace=True)
            df['Year'] = df[primary_date_col].dt.year
            print(f"Date range: {df[primary_date_col].min()} to {df[primary_date_col].max()}")
        fatality_cols = [col for col in df.columns if 'fatal' in col.lower()]
        if fatality_cols:
            for col in fatality_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['Total_Fatalities'] = df[fatality_cols].sum(axis=1)
            print(f"Total fatalities: {df['Total_Fatalities'].sum():,}")
    else:
        print("Consolidated dataframe 'df' not found or is empty.")

    # 3. Flight type classification
    def get_flight_type(row):
        operator = str(row['Operator']).lower()
        aircraft_type = str(row['Type']).lower()
        military_keywords = ['military', 'air force', 'army', 'navy', 'marine corps', 'coast guard']
        if any(keyword in operator for keyword in military_keywords) or any(keyword in aircraft_type for keyword in military_keywords):
            return 'Military'
        private_keywords = ['private', 'executive', 'charter', 'club', 'school', 'training']
        if any(keyword in operator for keyword in private_keywords):
            return 'Private'
        commercial_keywords = ['airlines', 'airways', 'aerolineas', 'air', 'lineas', 'avia', 'cargo']
        if any(keyword in operator for keyword in commercial_keywords):
            return 'Commercial'
        large_aircraft = ['boeing', 'airbus', 'douglas', 'lockheed', 'mcdonnell']
        if any(aircraft in aircraft_type for aircraft in large_aircraft):
            return 'Commercial'
        return 'Commercial'
    df['Flight_Type'] = df.apply(get_flight_type, axis=1)

    # 4. NLP-based crash cause analysis
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r"[^a-z\s']", "", text)
        text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
        return text
    df['Summary_Clean'] = df['Summary'].fillna('').astype(str).apply(clean_text)
    cause_keywords = {
        'Mechanical/Engine Failure': ['engin', 'failur', 'mechan', 'structur', 'malfunct', 'gear', 'propel', 'hydraul', 'fire', 'explos', 'break', 'crack', 'fatigue', 'maintenance', 'defect', 'component', 'system', 'turbine', 'compressor', 'combustion', 'oil', 'leak', 'pressure'],
        'Weather Related': ['weather', 'storm', 'ice', 'fog', 'wind', 'thunderstorm', 'turbul', 'snow', 'rain', 'hail', 'lightning', 'visibility', 'shear', 'downdraft', 'updraft', 'cyclone', 'squall', 'freezing', 'icing'],
        'Pilot Error/Human Factor': ['pilot', 'error', 'human', 'crew', 'procedur', 'navig', 'stall', 'altitud', 'captain', 'officer', 'mistake', 'judgment', 'training', 'experience', 'fatigue', 'distraction', 'communication', 'coordination', 'decision'],
        'Attack/Sabotage': ['shot', 'hijack', 'bomb', 'attack', 'terrorist', 'sabotag', 'missile', 'military', 'war', 'combat', 'hostile', 'enemy'],
        'Loss of Control': ['control', 'loss', 'uncontrol', 'dive', 'spin', 'spiral', 'unstable', 'recovery', 'nose', 'tail', 'bank', 'roll'],
        'Terrain Collision': ['terrain', 'mountain', 'hill', 'ground', 'cfit', 'impact', 'altitude', 'elevation', 'obstacle', 'trees', 'building'],
        'Fuel Issue': ['fuel', 'exhaust', 'starvat', 'empty', 'shortage', 'consumption', 'tank', 'pump', 'line', 'contamination'],
        'Bird Strike': ['bird', 'strike', 'wildlife', 'flock', 'geese', 'duck'],
        'Runway/Airport Issues': ['runway', 'airport', 'takeoff', 'landing', 'approach', 'taxi', 'ground', 'collision', 'overrun', 'undershoot']
    }
    def classify_all_causes_enhanced(summary):
        causes = []
        for cause, keywords in cause_keywords.items():
            if any(keyword in summary for keyword in keywords):
                causes.append(cause)
        if not causes:
            if any(word in summary for word in ['fell', 'drop', 'plunge', 'crash']):
                causes.append('Loss of Control')
            elif any(word in summary for word in ['fire', 'burn', 'smoke']):
                causes.append('Mechanical/Engine Failure')
            elif any(word in summary for word in ['collid', 'hit', 'struck']):
                causes.append('Terrain Collision')
            else:
                causes.append('Unknown/Other')
        return causes
    df['Crash_Causes'] = df['Summary_Clean'].apply(classify_all_causes_enhanced)

    # 5. (Optional) Add your visualizations, modeling, and output saving here
    # Example: plt.show() or plt.savefig('output.png')

    # 6. Create visualizations directory
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)

    # 7. Plot and save missing data visualization
    if not df.empty:
        plt.figure(figsize=(10, 6))
        missing_data.plot.bar(x='Column', y='Missing_Percentage', legend=False)
        plt.title('Missing Data Percentage by Column')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'missing_data.png'))
        plt.close()

    # 8. Plot and save flight type distribution
    if 'Flight_Type' in df.columns:
        flight_type_distribution = df['Flight_Type'].value_counts()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=flight_type_distribution.index, y=flight_type_distribution.values, palette='viridis')
        plt.title('Distribution of Aviation Incidents by Flight Type')
        plt.xlabel('Flight Type')
        plt.ylabel('Number of Incidents')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'flight_type_distribution.png'))
        plt.close()

    # 9. Plot and save crash cause distribution
    if 'Crash_Causes' in df.columns:
        all_causes = []
        for causes_list in df['Crash_Causes']:
            all_causes.extend(causes_list)
        cause_counts = Counter(all_causes)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(cause_counts.keys()), y=list(cause_counts.values()), palette='mako')
        plt.title('Distribution of Crash Causes')
        plt.xlabel('Crash Cause')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'crash_cause_distribution.png'))
        plt.close()

    # 10. Advanced visualizations (from notebook phase)
    # Cause co-occurrence heatmap and network graph for commercial flights
    # Filter for commercial flights only
    commercial_df = df[df['Flight_Type'] == 'Commercial'].copy()
    commercial_multi_cause = commercial_df[commercial_df['Crash_Causes'].apply(len) > 1]
    all_pairs = []
    for _, row in commercial_multi_cause.iterrows():
        pairs = list(itertools.combinations(sorted(row['Crash_Causes']), 2))
        all_pairs.extend(pairs)
    pair_counts = Counter(all_pairs)
    cause_list = sorted(list(set([c for c in cause_keywords.keys()] + [c for pair in pair_counts for c in pair])))
    co_occurrence_matrix = pd.DataFrame(0, index=cause_list, columns=cause_list)
    for (cause1, cause2), count in pair_counts.items():
        co_occurrence_matrix.loc[cause1, cause2] = count
        co_occurrence_matrix.loc[cause2, cause1] = count
    # Save heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="Reds", linewidths=.5)
    plt.title('Heatmap of Co-occurring Crash Causes (Commercial Flights)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cause_cooccurrence_heatmap.png'))
    plt.close()
    # Save network graph
    G = nx.Graph()
    for cause in cause_list:
        total_occurrences = sum(commercial_df['Crash_Causes'].apply(lambda x: cause in x))
        if total_occurrences > 0:
            G.add_node(cause, size=total_occurrences)
    for (cause1, cause2), weight in pair_counts.items():
        if weight > 0:
            G.add_edge(cause1, cause2, weight=weight)
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G, k=0.8, iterations=50)
        node_sizes = [d['size'] * 10 for n, d in G.nodes(data=True)]
        edge_widths = [d['weight'] * 0.5 for u, v, d in G.edges(data=True)]
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        plt.title('Network Graph of Crash Cause Interactions (Commercial Flights)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'cause_cooccurrence_network.png'))
        plt.close()
    # Commercial flight analysis dashboard (multi-panel)
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('Comprehensive Commercial Flight Analysis Dashboard', fontsize=20, fontweight='bold')
    # Plot 1: Cause Distribution
    commercial_causes = []
    for causes_list in commercial_df['Crash_Causes']:
        commercial_causes.extend(causes_list)
    commercial_cause_dist = Counter(commercial_causes)
    commercial_cause_names = list(commercial_cause_dist.keys())[:8]
    commercial_cause_counts = [commercial_cause_dist[cause] for cause in commercial_cause_names]
    axes[0, 0].barh(commercial_cause_names, commercial_cause_counts, color='steelblue')
    axes[0, 0].set_title('Commercial Flight Crash Causes', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Incidents')
    # Plot 2: Temporal Trends
    if 'Year' in commercial_df.columns:
        yearly_commercial = commercial_df.groupby('Year').size()
        axes[0, 1].plot(yearly_commercial.index, yearly_commercial.values, linewidth=2, color='darkblue')
        axes[0, 1].set_title('Commercial Incidents Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Incidents')
    # Plot 3: Severity Distribution
    if 'Fatalities' in commercial_df.columns:
        commercial_df['Severity_Category'] = pd.cut(commercial_df['Fatalities'], bins=[-1, 0, 5, 25, 100, float('inf')], labels=['No Fatalities', 'Minor (1-5)', 'Moderate (6-25)', 'Major (26-100)', 'Catastrophic (100+)'])
        severity_counts = commercial_df['Severity_Category'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        axes[0, 2].pie(severity_counts.values, labels=severity_counts.index, colors=colors[:len(severity_counts)], autopct='%1.1f%%')
        axes[0, 2].set_title('Commercial Flight Severity Distribution', fontweight='bold')
    # Plot 4: Top Operators
    if 'Operator' in commercial_df.columns:
        top_ops = commercial_df['Operator'].value_counts().head(6)
        axes[1, 0].bar(range(len(top_ops)), top_ops.values, color='lightcoral')
        axes[1, 0].set_xticks(range(len(top_ops)))
        axes[1, 0].set_xticklabels([op[:15] + '...' if len(op) > 15 else op for op in top_ops.index], rotation=45, ha='right')
        axes[1, 0].set_title('Top Commercial Operators', fontweight='bold')
        axes[1, 0].set_ylabel('Incidents')
    # Plot 5: Aircraft Manufacturers
    if 'Type' in commercial_df.columns:
        commercial_df['Manufacturer'] = commercial_df['Type'].str.extract(r'^([A-Za-z]+)')[0]
        manufacturer_dist = commercial_df['Manufacturer'].value_counts().head(8)
        axes[1, 1].bar(manufacturer_dist.index, manufacturer_dist.values, color='lightgreen')
        axes[1, 1].set_title('Aircraft Manufacturers', fontweight='bold')
        axes[1, 1].set_xlabel('Manufacturer')
        axes[1, 1].set_ylabel('Incidents')
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    # Plot 6: Decade Comparison
    if 'Year' in commercial_df.columns:
        commercial_df['Decade'] = (commercial_df['Year'] // 10) * 10
        decade_incidents = commercial_df.groupby('Decade').size()
        axes[1, 2].bar(decade_incidents.index, decade_incidents.values, color='gold', width=8)
        axes[1, 2].set_title('Commercial Incidents by Decade', fontweight='bold')
        axes[1, 2].set_xlabel('Decade')
        axes[1, 2].set_ylabel('Incidents')
    # Plot 7: Cause vs Severity Heatmap
    if 'Severity_Category' in commercial_df.columns and len(commercial_causes) > 0:
        cause_severity_matrix = pd.DataFrame()
        for cause in commercial_cause_dist.most_common(6):
            cause_name = cause[0]
            cause_incidents = commercial_df[commercial_df['Crash_Causes'].apply(lambda x: cause_name in x)]
            severity_counts = cause_incidents['Severity_Category'].value_counts()
            cause_severity_matrix[cause_name] = severity_counts
        cause_severity_matrix = cause_severity_matrix.fillna(0)
        sns.heatmap(cause_severity_matrix, annot=True, fmt='.0f', cmap='Reds', ax=axes[2, 0])
        axes[2, 0].set_title('Cause vs Severity Matrix', fontweight='bold')
        plt.setp(axes[2, 0].get_xticklabels(), rotation=45, ha='right')
    # Plot 8: Fatalities Distribution
    if 'Fatalities' in commercial_df.columns:
        fatalities_data = commercial_df['Fatalities'].dropna()
        axes[2, 1].hist(fatalities_data[fatalities_data <= 300], bins=30, color='purple', alpha=0.7)
        axes[2, 1].set_title('Commercial Flight Fatalities Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('Number of Fatalities')
        axes[2, 1].set_ylabel('Frequency')
    # Plot 9: Monthly Trends
    if 'Date' in commercial_df.columns:
        try:
            commercial_df['Month'] = pd.to_datetime(commercial_df['Date'], errors='coerce').dt.month
            monthly_incidents = commercial_df.groupby('Month').size()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[2, 2].plot(range(1, 13), [monthly_incidents.get(i, 0) for i in range(1, 13)], marker='o', linewidth=2, color='teal')
            axes[2, 2].set_xticks(range(1, 13))
            axes[2, 2].set_xticklabels(months, rotation=45)
            axes[2, 2].set_title('Commercial Incidents by Month', fontweight='bold')
            axes[2, 2].set_ylabel('Incidents')
        except:
            axes[2, 2].text(0.5, 0.5, 'Date data not available', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Monthly Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'commercial_flight_dashboard.png'))
    plt.close()

    # 11. Machine Learning Pipeline: Predicting Crash Severity (Binary Classification)
    # Prepare data for ML (binary target: Severe vs. Not Severe)
    # Define 'Severe' as incidents with Fatalities >= 1
    df_ml = df.copy()
    if 'Fatalities' in df_ml.columns:
        df_ml['Severe'] = (df_ml['Fatalities'] >= 1).astype(int)
    else:
        print('Fatalities column not found. Skipping ML.')
        return
    # Drop columns with >80% missing data
    missing_threshold = 80
    cols_to_drop = [col for col in df_ml.columns if df_ml[col].isnull().mean() * 100 > missing_threshold]
    df_ml = df_ml.drop(columns=cols_to_drop)

    # Group rare categories for Operator and Type
    for cat_col in ['Operator', 'Type']:
        if cat_col in df_ml.columns:
            top_cats = df_ml[cat_col].value_counts().nlargest(10).index
            df_ml[cat_col] = df_ml[cat_col].apply(lambda x: x if x in top_cats else 'Other')

    # Impute missing values: median for numeric, mode (first value) for categorical (only if scalar)
    for col in df_ml.columns:
        if df_ml[col].dtype in [np.float64, np.int64]:
            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())
        elif df_ml[col].dtype == object:
            mode_val = df_ml[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            # Only fill if fill_val is a scalar (not a list, dict, etc.)
            if not isinstance(fill_val, (list, dict, set)):
                df_ml[col] = df_ml[col].fillna(fill_val)

    # Add 'Is_Recent' feature
    if 'Year' in df_ml.columns:
        df_ml['Is_Recent'] = (df_ml['Year'] > 2000).astype(int)

    # Feature selection: use more informative features
    feature_cols = []
    if 'Year' in df_ml.columns:
        feature_cols.append('Year')
    if 'Aboard' in df_ml.columns:
        feature_cols.append('Aboard')
    for col in [
        'Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',
        'Control_Metric', 'Turbulence_In_gforces', 'Cabin_Temperature',
        'Max_Elevation', 'Violations', 'Adverse_Weather_Metric', 'Is_Recent']:
        if col in df_ml.columns:
            feature_cols.append(col)
    # Encoded categorical features
    if 'Flight_Type' in df_ml.columns:
        df_ml['Flight_Type_Code'] = df_ml['Flight_Type'].astype('category').cat.codes
        feature_cols.append('Flight_Type_Code')
    if 'Operator' in df_ml.columns:
        df_ml['Operator_Code'] = df_ml['Operator'].astype('category').cat.codes
        feature_cols.append('Operator_Code')
    if 'Type' in df_ml.columns:
        df_ml['Type_Code'] = df_ml['Type'].astype('category').cat.codes
        feature_cols.append('Type_Code')
    if not feature_cols:
        print('No usable features for ML. Skipping model training.')
        return
    X = df_ml[feature_cols]
    y = df_ml['Severe']

    # Scale numeric features for XGBoost
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled, X_test_scaled = train_test_split(X_scaled, test_size=0.2, random_state=42, stratify=y)

    # Random Forest with class_weight
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    print(f'Random Forest Accuracy: {acc_rf:.3f}, F1 Score: {f1_rf:.3f}')
    # XGBoost with scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
    print(f'XGBoost Accuracy: {acc_xgb:.3f}, F1 Score: {f1_xgb:.3f}')
    # Classification report and confusion matrix (for Random Forest)
    print('\nRandom Forest Classification Report:')
    print(classification_report(y_test, y_pred_rf))
    print('Random Forest Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_rf))
    # Save confusion matrix as heatmap
    cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Severe', 'Severe'], yticklabels=['Not Severe', 'Severe'])
    plt.title('Random Forest Confusion Matrix (Binary)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rf_confusion_matrix_binary.png'))
    plt.close()
    # Model accuracy comparison bar plot
    model_names = ['Random Forest', 'XGBoost']
    accuracies = [acc_rf, acc_xgb]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=model_names, y=accuracies, palette='Set2')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison (Binary)')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'model_accuracy_comparison_binary.png'))
    plt.close()
    # SHAP Analysis (XAI) for Random Forest and XGBoost
    print("\nRunning SHAP (XAI) analysis...")
    shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_sample_scaled = X_test_scaled.sample(min(200, len(X_test_scaled)), random_state=42)
    # Random Forest SHAP
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(shap_sample)
    if isinstance(shap_values_rf, list) and len(shap_values_rf) == 2:
        shap_values_rf_plot = shap_values_rf[1] - shap_values_rf[0]
    elif isinstance(shap_values_rf, np.ndarray):
        shap_values_rf_plot = shap_values_rf
    else:
        shap_values_rf_plot = shap_values_rf[1] if isinstance(shap_values_rf, list) else shap_values_rf
    plt.figure()
    shap.summary_plot(shap_values_rf_plot, shap_sample, feature_names=feature_cols, show=False)
    plt.title('SHAP Summary (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'shap_summary_rf.png'))
    plt.close()
    # XGBoost SHAP
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(shap_sample_scaled)
    plt.figure()
    shap.summary_plot(shap_values_xgb, shap_sample_scaled, feature_names=feature_cols, show=False)
    plt.title('SHAP Summary (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'shap_summary_xgb.png'))
    plt.close()
    print("SHAP summary plots saved in visualizations/.")
    print(f"\nVisualizations saved in: {vis_dir}")
    print("\nScript execution complete. All analysis performed in pure Python.")

    # --- Experimental: Remove 'Aboard' and 'Year' for model training and SHAP analysis ---
    print("\n[Experimental] Training models with 'Aboard' and 'Year' REMOVED from features...")
    reduced_feature_cols = [col for col in feature_cols if col not in ['Aboard', 'Year']]
    if not reduced_feature_cols:
        print('No usable features after removing Aboard and Year. Skipping experimental section.')
    else:
        X_reduced = df_ml[reduced_feature_cols]
        # Scale numeric features for XGBoost
        X_reduced_scaled = X_reduced.copy()
        num_cols_reduced = X_reduced.select_dtypes(include=[np.number]).columns
        X_reduced_scaled[num_cols_reduced] = scaler.fit_transform(X_reduced[num_cols_reduced])
        # Train/test split
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
        X_train_r_scaled, X_test_r_scaled = train_test_split(X_reduced_scaled, test_size=0.2, random_state=42, stratify=y)
        # Random Forest
        rf_r = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_r.fit(X_train_r, y_train_r)
        y_pred_rf_r = rf_r.predict(X_test_r)
        acc_rf_r = accuracy_score(y_test_r, y_pred_rf_r)
        print(f"[Experimental] Random Forest Accuracy (no 'Aboard'/'Year'): {acc_rf_r:.3f}")
        # XGBoost
        scale_pos_weight_r = (y_train_r == 0).sum() / (y_train_r == 1).sum()
        xgb_r = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_r)
        xgb_r.fit(X_train_r_scaled, y_train_r)
        y_pred_xgb_r = xgb_r.predict(X_test_r_scaled)
        acc_xgb_r = accuracy_score(y_test_r, y_pred_xgb_r)
        print(f"[Experimental] XGBoost Accuracy (no 'Aboard'/'Year'): {acc_xgb_r:.3f}")
        # SHAP for Random Forest
        shap_sample_r = X_test_r.sample(min(200, len(X_test_r)), random_state=42)
        explainer_rf_r = shap.TreeExplainer(rf_r)
        shap_values_rf_r = explainer_rf_r.shap_values(shap_sample_r)
        if isinstance(shap_values_rf_r, list) and len(shap_values_rf_r) == 2:
            shap_values_rf_r_plot = shap_values_rf_r[1] - shap_values_rf_r[0]
        elif isinstance(shap_values_rf_r, np.ndarray):
            shap_values_rf_r_plot = shap_values_rf_r
        else:
            shap_values_rf_r_plot = shap_values_rf_r[1] if isinstance(shap_values_rf_r, list) else shap_values_rf_r
        plt.figure()
        shap.summary_plot(shap_values_rf_r_plot, shap_sample_r, feature_names=reduced_feature_cols, show=False)
        plt.title("SHAP Summary (Random Forest, no 'Aboard'/'Year')")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_rf_no_aboard_year.png'))
        plt.close()
        # SHAP for XGBoost
        shap_sample_r_scaled = X_test_r_scaled.sample(min(200, len(X_test_r_scaled)), random_state=42)
        explainer_xgb_r = shap.TreeExplainer(xgb_r)
        shap_values_xgb_r = explainer_xgb_r.shap_values(shap_sample_r_scaled)
        plt.figure()
        shap.summary_plot(shap_values_xgb_r, shap_sample_r_scaled, feature_names=reduced_feature_cols, show=False)
        plt.title("SHAP Summary (XGBoost, no 'Aboard'/'Year')")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_xgb_no_aboard_year.png'))
        plt.close()
        print("[Experimental] SHAP summary plots (no 'Aboard'/'Year') saved in visualizations/.")
    # --- Experimental: Interpretable Features Only (Dashboard Features) ---
    print("\n[Experimental] Training models using only interpretable dashboard features...")
    # Prepare interpretable features
    df_ml['Main_Cause'] = df_ml['Crash_Causes'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown/Other')
    if 'Type' in df_ml.columns:
        df_ml['Manufacturer'] = df_ml['Type'].str.extract(r'^([A-Za-z]+)')[0].fillna('Unknown')
    interpretable_features = []
    # Add main cause (label encoded)
    if 'Main_Cause' in df_ml.columns:
        df_ml['Main_Cause_Code'] = df_ml['Main_Cause'].astype('category').cat.codes
        interpretable_features.append('Main_Cause_Code')
    # Add operator (label encoded)
    if 'Operator' in df_ml.columns:
        df_ml['Operator_Code'] = df_ml['Operator'].astype('category').cat.codes
        interpretable_features.append('Operator_Code')
    # Add manufacturer (label encoded)
    if 'Manufacturer' in df_ml.columns:
        df_ml['Manufacturer_Code'] = df_ml['Manufacturer'].astype('category').cat.codes
        interpretable_features.append('Manufacturer_Code')
    # Add flight type (label encoded)
    if 'Flight_Type' in df_ml.columns:
        df_ml['Flight_Type_Code'] = df_ml['Flight_Type'].astype('category').cat.codes
        interpretable_features.append('Flight_Type_Code')
    # Prepare X and y
    if not interpretable_features:
        print('No interpretable features found. Skipping this experiment.')
    else:
        X_interpret = df_ml[interpretable_features]
        y_interpret = df_ml['Severe']
        # Train/test split
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_interpret, y_interpret, test_size=0.2, random_state=42, stratify=y_interpret)
        # Random Forest
        rf_i = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_i.fit(X_train_i, y_train_i)
        y_pred_rf_i = rf_i.predict(X_test_i)
        acc_rf_i = accuracy_score(y_test_i, y_pred_rf_i)
        print(f"[Experimental] Random Forest Accuracy (interpretable features): {acc_rf_i:.3f}")
        # XGBoost
        scale_pos_weight_i = (y_train_i == 0).sum() / (y_train_i == 1).sum()
        xgb_i = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_i)
        xgb_i.fit(X_train_i, y_train_i)
        y_pred_xgb_i = xgb_i.predict(X_test_i)
        acc_xgb_i = accuracy_score(y_test_i, y_pred_xgb_i)
        print(f"[Experimental] XGBoost Accuracy (interpretable features): {acc_xgb_i:.3f}")
        # SHAP for Random Forest
        shap_sample_i = X_test_i.sample(min(200, len(X_test_i)), random_state=42)
        explainer_rf_i = shap.TreeExplainer(rf_i)
        shap_values_rf_i = explainer_rf_i.shap_values(shap_sample_i)
        if isinstance(shap_values_rf_i, list) and len(shap_values_rf_i) == 2:
            shap_values_rf_i_plot = shap_values_rf_i[1] - shap_values_rf_i[0]
        elif isinstance(shap_values_rf_i, np.ndarray):
            shap_values_rf_i_plot = shap_values_rf_i
        else:
            shap_values_rf_i_plot = shap_values_rf_i[1] if isinstance(shap_values_rf_i, list) else shap_values_rf_i
        plt.figure()
        shap.summary_plot(shap_values_rf_i_plot, shap_sample_i, feature_names=interpretable_features, show=False)
        plt.title("SHAP Summary (Random Forest, interpretable features)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_rf_interpretable.png'))
        plt.close()
        # SHAP for XGBoost
        shap_sample_xgb_i = X_test_i.sample(min(200, len(X_test_i)), random_state=42)
        explainer_xgb_i = shap.TreeExplainer(xgb_i)
        shap_values_xgb_i = explainer_xgb_i.shap_values(shap_sample_xgb_i)
        plt.figure()
        shap.summary_plot(shap_values_xgb_i, shap_sample_xgb_i, feature_names=interpretable_features, show=False)
        plt.title("SHAP Summary (XGBoost, interpretable features)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_xgb_interpretable.png'))
        plt.close()
        print("[Experimental] SHAP summary plots (interpretable features) saved in visualizations/.")
    # --- Experimental: Strict Dashboard Features Only ---
    print("\n[Experimental] Training models using ONLY dashboard-visualized features (cause, operator, manufacturer, severity category, flight type)...")
    # Prepare dashboard features
    df_ml['Main_Cause'] = df_ml['Crash_Causes'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown/Other')
    if 'Type' in df_ml.columns:
        df_ml['Manufacturer'] = df_ml['Type'].str.extract(r'^([A-Za-z]+)')[0].fillna('Unknown')
    if 'Fatalities' in df_ml.columns:
        df_ml['Severity_Category'] = pd.cut(df_ml['Fatalities'], bins=[-1, 0, 5, 25, 100, float('inf')], labels=['No Fatalities', 'Minor (1-5)', 'Moderate (6-25)', 'Major (26-100)', 'Catastrophic (100+)'])
    dashboard_features = []
    # Main cause (label encoded)
    if 'Main_Cause' in df_ml.columns:
        df_ml['Main_Cause_Code'] = df_ml['Main_Cause'].astype('category').cat.codes
        dashboard_features.append('Main_Cause_Code')
    # Operator (label encoded)
    if 'Operator' in df_ml.columns:
        df_ml['Operator_Code'] = df_ml['Operator'].astype('category').cat.codes
        dashboard_features.append('Operator_Code')
    # Manufacturer (label encoded)
    if 'Manufacturer' in df_ml.columns:
        df_ml['Manufacturer_Code'] = df_ml['Manufacturer'].astype('category').cat.codes
        dashboard_features.append('Manufacturer_Code')
    # Flight type (label encoded)
    if 'Flight_Type' in df_ml.columns:
        df_ml['Flight_Type_Code'] = df_ml['Flight_Type'].astype('category').cat.codes
        dashboard_features.append('Flight_Type_Code')
    # Prepare X and y
    if not dashboard_features:
        print('No dashboard features found. Skipping this experiment.')
    else:
        X_dash = df_ml[dashboard_features]
        y_dash = df_ml['Severe']
        # Train/test split
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dash, y_dash, test_size=0.2, random_state=42, stratify=y_dash)
        # Random Forest
        rf_d = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_d.fit(X_train_d, y_train_d)
        y_pred_rf_d = rf_d.predict(X_test_d)
        acc_rf_d = accuracy_score(y_test_d, y_pred_rf_d)
        print(f"[Experimental] Random Forest Accuracy (dashboard features): {acc_rf_d:.3f}")
        # XGBoost
        scale_pos_weight_d = (y_train_d == 0).sum() / (y_train_d == 1).sum()
        xgb_d = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight_d)
        xgb_d.fit(X_train_d, y_train_d)
        y_pred_xgb_d = xgb_d.predict(X_test_d)
        acc_xgb_d = accuracy_score(y_test_d, y_pred_xgb_d)
        print(f"[Experimental] XGBoost Accuracy (dashboard features): {acc_xgb_d:.3f}")
        # SHAP for Random Forest
        shap_sample_d = X_test_d.sample(min(200, len(X_test_d)), random_state=42)
        explainer_rf_d = shap.TreeExplainer(rf_d)
        shap_values_rf_d = explainer_rf_d.shap_values(shap_sample_d)
        if isinstance(shap_values_rf_d, list) and len(shap_values_rf_d) == 2:
            shap_values_rf_d_plot = shap_values_rf_d[1] - shap_values_rf_d[0]
        elif isinstance(shap_values_rf_d, np.ndarray):
            shap_values_rf_d_plot = shap_values_rf_d
        else:
            shap_values_rf_d_plot = shap_values_rf_d[1] if isinstance(shap_values_rf_d, list) else shap_values_rf_d
        plt.figure()
        shap.summary_plot(shap_values_rf_d_plot, shap_sample_d, feature_names=dashboard_features, show=False)
        plt.title("SHAP Summary (Random Forest, dashboard features)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_rf_dashboard.png'))
        plt.close()
        # SHAP for XGBoost
        shap_sample_xgb_d = X_test_d.sample(min(200, len(X_test_d)), random_state=42)
        explainer_xgb_d = shap.TreeExplainer(xgb_d)
        shap_values_xgb_d = explainer_xgb_d.shap_values(shap_sample_xgb_d)
        plt.figure()
        shap.summary_plot(shap_values_xgb_d, shap_sample_xgb_d, feature_names=dashboard_features, show=False)
        plt.title("SHAP Summary (XGBoost, dashboard features)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'shap_summary_xgb_dashboard.png'))
        plt.close()
        print("[Experimental] SHAP summary plots (dashboard features) saved in visualizations/.")

if __name__ == "__main__":
    main()
